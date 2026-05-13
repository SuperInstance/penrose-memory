//! Tensor-valued Penrose tiling — reverse actualization through the lattice.
//!
//! Each tile carries a rank-2 tensor whose shape is determined by tile type.
//! The five source coordinates (a–e) control orthogonal basis functions
//! (fill modes) that generate the tensor's values.

use crate::cut_and_project::{TileCoord, TileType};

/// A tensor-valued Penrose tile — carries a 2D tensor field on each tile.
#[derive(Debug, Clone)]
pub struct TensorTile {
    /// (x, y) position in the Penrose floor.
    pub position: [f64; 2],
    /// Rotation in radians.
    pub orientation: f64,
    /// Thick or Thin rhomb.
    pub tile_type: TileType,
    /// 5D lattice coordinates.
    pub source_coords: [i32; 5],
    /// Flat tensor data (row-major).
    pub tensor: Vec<f32>,
    /// (rows, cols) — varies by tile type.
    pub tensor_shape: (usize, usize),
}

impl TensorTile {
    /// Create a new tensor tile with the tensor initialized to zeros.
    ///
    /// - Thick tiles get shape `(5, 5)` to match the 5D source.
    /// - Thin tiles get shape `(3, 8)` so the proportion 8/3 ≈ φ.
    pub fn new(
        source_coords: [i32; 5],
        tile_type: TileType,
        orientation: f64,
        position: [f64; 2],
    ) -> Self {
        let shape = match tile_type {
            TileType::Thick => (5, 5),
            TileType::Thin => (3, 8),
            TileType::Rejected => (1, 1), // fallback — shouldn't appear in output
        };
        let len = shape.0 * shape.1;
        Self {
            position,
            orientation,
            tile_type,
            source_coords,
            tensor: vec![0.0f32; len],
            tensor_shape: shape,
        }
    }

    /// Fill the tensor using five orthogonal basis modes controlled by the
    /// five source coordinates.
    ///
    /// | coord | mode | description |
    /// |-------|------|-------------|
    /// | a (0) | constant | base intensity |
    /// | b (1) | linear gradient along rows | |
    /// | c (2) | sinusoidal along columns | |
    /// | d (3) | pseudo-random texture seed | |
    /// | e (4) | phase shift | |
    pub fn fill_from_source(&mut self) {
        let (rows, cols) = self.tensor_shape;
        let src = self.source_coords;

        // Normalize source coordinates to [0, 1] range.
        let a = (src[0].unsigned_abs() as f32).max(1.0) / 100.0;
        let b = src[1] as f32;
        let c = (src[2].unsigned_abs() as f32).max(1.0);
        let d = src[3].unsigned_abs() as u32;
        let e = src[4] as f32;

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let i_f = i as f32;
                let j_f = j as f32;
                let m = rows as f32;
                let n = cols as f32;

                // Mode 0 (a): constant fill based on source[0] normalized.
                let mode_a = a * (src[0].unsigned_abs() as f32 + 1.0) / 10.0;

                // Mode 1 (b): linear gradient along rows based on source[1].
                let mode_b = b * i_f / m;

                // Mode 2 (c): sinusoidal along columns based on source[2].
                let mode_c = (2.0 * std::f32::consts::PI * c * j_f / n).sin();

                // Mode 3 (d): pseudo-random seed based on source[3].
                let hash = Self::simple_hash(i as u32, j as u32, d);
                let mode_d = (hash as f32) / (u32::MAX as f32);

                // Mode 4 (e): phase shift based on source[4].
                let mode_e = (2.0 * std::f32::consts::PI * c * i_f / m + e / 10.0).sin();

                // Combine all modes additively.
                self.tensor[idx] = mode_a + mode_b + mode_c + mode_d + mode_e;
            }
        }
    }

    /// Simple deterministic hash for pseudo-random texture generation.
    fn simple_hash(i: u32, j: u32, seed: u32) -> u32 {
        // FNV-1a–inspired mixing
        let mut h = 2166136261u32;
        h ^= i.wrapping_mul(seed.wrapping_add(1));
        h = h.wrapping_mul(16777619);
        h ^= j.wrapping_mul(seed.wrapping_add(7));
        h = h.wrapping_mul(16777619);
        h ^= seed;
        h = h.wrapping_mul(16777619);
        h
    }

    /// Zero out tensor values below the given threshold.
    pub fn apply_threshold(&mut self, threshold: f32) {
        for v in &mut self.tensor {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    /// Index into the flat tensor at (row, col).
    pub fn tensor_at(&self, row: usize, col: usize) -> f32 {
        let (rows, cols) = self.tensor_shape;
        assert!(row < rows && col < cols, "index out of bounds");
        self.tensor[row * cols + col]
    }

    /// L1 norm — sum of absolute values (constraint energy).
    pub fn l1_norm(&self) -> f32 {
        self.tensor.iter().map(|v| v.abs()).sum()
    }

    /// L2 norm — Euclidean norm (signal strength).
    pub fn l2_norm(&self) -> f32 {
        let sum_sq: f32 = self.tensor.iter().map(|v| v * v).sum();
        sum_sq.sqrt()
    }

    /// Number of elements in the tensor.
    pub fn tensor_len(&self) -> usize {
        self.tensor_shape.0 * self.tensor_shape.1
    }
}

// ──────────────────────────────────────────────────────────────────────
// TensorTiling — collection of tensor tiles with adjacency info
// ──────────────────────────────────────────────────────────────────────

/// A complete tensor-valued tiling with adjacency information.
#[derive(Debug, Clone)]
pub struct TensorTiling {
    pub tiles: Vec<TensorTile>,
    /// (tile_i, tile_j, shared_edge_orientation in radians).
    pub adjacency: Vec<(usize, usize, f64)>,
}

impl TensorTiling {
    /// Create a new tiling, auto-detecting adjacency from tile positions.
    ///
    /// Two tiles are considered adjacent if their centres are within
    /// `2.0` units of each other (typical Penrose edge length ≈ 1.0).
    pub fn new(tiles: Vec<TensorTile>) -> Self {
        let adjacency = Self::detect_adjacency(&tiles);
        Self { tiles, adjacency }
    }

    /// Detect adjacency: tiles within threshold distance share an edge.
    fn detect_adjacency(tiles: &[TensorTile]) -> Vec<(usize, usize, f64)> {
        let threshold = 2.0;
        let threshold_sq = threshold * threshold;
        let mut edges = Vec::new();

        for i in 0..tiles.len() {
            for j in (i + 1)..tiles.len() {
                let dx = tiles[i].position[0] - tiles[j].position[0];
                let dy = tiles[i].position[1] - tiles[j].position[1];
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < threshold_sq {
                    // Shared edge orientation = angle of the line connecting centres.
                    let orientation = dy.atan2(dx);
                    edges.push((i, j, orientation));
                }
            }
        }
        edges
    }

    /// Apply a function to every tile in the tiling.
    pub fn apply_kernel<F>(&mut self, f: F)
    where
        F: Fn(&mut TensorTile),
    {
        for tile in &mut self.tiles {
            f(tile);
        }
    }

    /// Constraint check: sum of L1 border mismatches between adjacent tiles.
    ///
    /// For each pair of adjacent tiles, compare the last row/first row (or
    /// last col/first col depending on orientation). The mismatch is the L1
    /// distance between these border vectors.
    pub fn constraint_check(&self) -> f32 {
        let mut total_mismatch = 0.0f32;

        for &(i, j, _orientation) in &self.adjacency {
            let tile_a = &self.tiles[i];
            let tile_b = &self.tiles[j];
            let (rows_a, cols_a) = tile_a.tensor_shape;
            let (rows_b, cols_b) = tile_b.tensor_shape;

            // Use the narrower dimension for comparison.
            let border_len = cols_a.min(cols_b);

            // Compare tile A's last row with tile B's first row.
            for col in 0..border_len {
                let va = tile_a.tensor_at(rows_a - 1, col.min(cols_a - 1));
                let vb = tile_b.tensor_at(0, col.min(cols_b - 1));
                total_mismatch += (va - vb).abs();
            }

            // Also compare first columns if shapes allow.
            let col_border_len = rows_a.min(rows_b);
            for row in 0..col_border_len {
                let va = tile_a.tensor_at(row.min(rows_a - 1), cols_a - 1);
                let vb = tile_b.tensor_at(row.min(rows_b - 1), 0);
                total_mismatch += (va - vb).abs();
            }
        }

        total_mismatch
    }
}

// ──────────────────────────────────────────────────────────────────────
// Top-level generation function
// ──────────────────────────────────────────────────────────────────────

/// Generate a tensor tiling from 5D lattice points and their projected tiles.
///
/// Takes 5D lattice points and baseline `TileCoord`s (from the cut-and-project
/// compiler), lifts them into `TensorTile`s with filled tensors.
pub fn generate_tensor_tiling(
    _lattice_points: &[[i32; 5]],
    baseline_tiles: &[TileCoord],
) -> TensorTiling {
    let mut tensor_tiles = Vec::with_capacity(baseline_tiles.len());

    for tc in baseline_tiles {
        // Convert source_coords Vec<i32> to [i32; 5].
        let mut coords = [0i32; 5];
        for (k, &v) in tc.source_coords.iter().enumerate().take(5) {
            coords[k] = v;
        }

        let mut tt = TensorTile::new(
            coords,
            tc.tile_type,
            0.0, // orientation derived from position for now
            [tc.x, tc.y],
        );
        tt.fill_from_source();
        tensor_tiles.push(tt);
    }

    TensorTiling::new(tensor_tiles)
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut_and_project::{CutAndProjectCompiler, TileType};

    #[test]
    fn test_tensor_tile_creation() {
        let tile = TensorTile::new(
            [1, 2, 3, 4, 5],
            TileType::Thick,
            0.0,
            [1.0, 2.0],
        );
        assert_eq!(tile.tensor_shape, (5, 5));
        assert_eq!(tile.tensor.len(), 25);
        // All zeros initially.
        assert!(tile.tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_tile_creation_thin() {
        let tile = TensorTile::new(
            [1, 2, 3, 4, 5],
            TileType::Thin,
            0.0,
            [0.0, 0.0],
        );
        assert_eq!(tile.tensor_shape, (3, 8));
        assert_eq!(tile.tensor.len(), 24);
    }

    #[test]
    fn test_tensor_tile_fill_modes() {
        // Create five tiles varying only one source coordinate at a time.
        // Each should produce different tensors.
        let bases: [[i32; 5]; 5] = [
            [10, 0, 0, 0, 0],
            [0, 10, 0, 0, 0],
            [0, 0, 10, 0, 0],
            [0, 0, 0, 10, 0],
            [0, 0, 0, 0, 10],
        ];

        let filled: Vec<Vec<f32>> = bases
            .iter()
            .map(|&coords| {
                let mut tile =
                    TensorTile::new(coords, TileType::Thick, 0.0, [0.0, 0.0]);
                tile.fill_from_source();
                tile.tensor.clone()
            })
            .collect();

        // Each pair of fill modes should differ.
        for i in 0..filled.len() {
            for j in (i + 1)..filled.len() {
                assert_ne!(
                    filled[i], filled[j],
                    "Fill modes {} and {} produced identical tensors",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_threshold_filter() {
        let mut tile = TensorTile::new(
            [5, 5, 5, 5, 5],
            TileType::Thick,
            0.0,
            [0.0, 0.0],
        );
        tile.fill_from_source();

        // Remember which values were above/below 0.5.
        let before = tile.tensor.clone();
        tile.apply_threshold(0.5);

        // Every value below 0.5 should now be 0.
        for (idx, &v) in tile.tensor.iter().enumerate() {
            if before[idx] < 0.5 {
                assert_eq!(v, 0.0, "value {} was below threshold but not zeroed", before[idx]);
            } else {
                assert_eq!(v, before[idx], "value {} was above threshold but changed", before[idx]);
            }
        }
    }

    #[test]
    fn test_tiling_adjacency() {
        // Two tiles very close together should be detected as adjacent.
        let tiles = vec![
            TensorTile::new([1, 0, 0, 0, 0], TileType::Thick, 0.0, [0.0, 0.0]),
            TensorTile::new([0, 1, 0, 0, 0], TileType::Thick, 0.0, [0.5, 0.5]),
        ];
        let tiling = TensorTiling::new(tiles);
        assert!(
            !tiling.adjacency.is_empty(),
            "Two close tiles should detect adjacency"
        );
    }

    #[test]
    fn test_tiling_no_adjacency_far_tiles() {
        // Two tiles far apart should NOT be adjacent.
        let tiles = vec![
            TensorTile::new([1, 0, 0, 0, 0], TileType::Thick, 0.0, [0.0, 0.0]),
            TensorTile::new([0, 1, 0, 0, 0], TileType::Thick, 0.0, [100.0, 100.0]),
        ];
        let tiling = TensorTiling::new(tiles);
        assert!(
            tiling.adjacency.is_empty(),
            "Far tiles should not be adjacent"
        );
    }

    #[test]
    fn test_constraint_check_identical_vs_different() {
        // Two identical tiles: low mismatch.
        let mut tile_a =
            TensorTile::new([2, 2, 2, 2, 2], TileType::Thick, 0.0, [0.0, 0.0]);
        tile_a.fill_from_source();

        // Clone for identical comparison.
        let mut tile_b_identical = tile_a.clone();
        tile_b_identical.position = [0.5, 0.5]; // close enough for adjacency

        let tiling_identical = TensorTiling::new(vec![tile_a.clone(), tile_b_identical]);
        let mismatch_identical = tiling_identical.constraint_check();

        // Two very different tiles: higher mismatch.
        let mut tile_c =
            TensorTile::new([50, 50, 50, 50, 50], TileType::Thick, 0.0, [0.0, 0.0]);
        tile_c.fill_from_source();
        let mut tile_d = tile_c.clone();
        tile_d.position = [0.5, 0.5];

        let tiling_different = TensorTiling::new(vec![tile_a, tile_d]);
        let mismatch_different = tiling_different.constraint_check();

        // Different tiles should have mismatch ≥ identical tiles.
        assert!(
            mismatch_different >= mismatch_identical,
            "Different tiles (mismatch={}) should have >= mismatch than identical ({})",
            mismatch_different,
            mismatch_identical,
        );
    }

    #[test]
    fn test_norms() {
        let mut tile =
            TensorTile::new([3, 3, 3, 3, 3], TileType::Thick, 0.0, [0.0, 0.0]);
        tile.fill_from_source();
        let l1 = tile.l1_norm();
        let l2 = tile.l2_norm();
        assert!(l1 > 0.0, "L1 norm should be positive after fill");
        assert!(l2 > 0.0, "L2 norm should be positive after fill");
        // L1 ≥ L2 always holds.
        assert!(l1 >= l2, "L1 ({}) should be >= L2 ({})", l1, l2);
    }

    #[test]
    fn test_generate_tensor_tiling() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let baseline = compiler.compile(2);
        if baseline.is_empty() {
            return; // range too small to get tiles — not an error
        }
        let lattice: Vec<[i32; 5]> = baseline
            .iter()
            .map(|tc| {
                let mut c = [0i32; 5];
                for (k, &v) in tc.source_coords.iter().enumerate().take(5) {
                    c[k] = v;
                }
                c
            })
            .collect();
        let tiling = generate_tensor_tiling(&lattice, &baseline);
        assert_eq!(tiling.tiles.len(), baseline.len());
        // All tensors should be filled (not all zeros).
        let any_filled = tiling
            .tiles
            .iter()
            .any(|t| t.tensor.iter().any(|&v| v != 0.0));
        assert!(any_filled, "At least one tensor should have non-zero values");
    }
}
