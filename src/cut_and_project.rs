//! Generalized cut-and-project compiler.
//!
//! Takes a high-dimensional lattice, defines an acceptance window in
//! perpendicular space, and projects accepted lattice points to a
//! lower-dimensional aperiodic tiling. The classic Penrose construction
//! maps 5D → 2D via golden-angle rotations.

const PHI: f64 = 1.618033988749895;
const INV_PHI: f64 = 0.618033988749895;
const GOLDEN_ANGLE_RAD: f64 = 2.0 * std::f64::consts::PI / (PHI * PHI); // ≈ 2.399…

/// Tile type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    /// Thick (wide) rhomb — interior angle 72°.
    Thick,
    /// Thin (narrow) rhomb — interior angle 36°.
    Thin,
    /// Acceptance window rejected this point (should not appear in output).
    Rejected,
}

/// Coordinates of a single projected tile.
#[derive(Debug, Clone)]
pub struct TileCoord {
    /// Projected x in target (2D) space.
    pub x: f64,
    /// Projected y in target (2D) space.
    pub y: f64,
    /// Original lattice coordinates in source space.
    pub source_coords: Vec<i32>,
    /// Classified tile type.
    pub tile_type: TileType,
}

/// Report from Penrose verification.
#[derive(Debug, Clone)]
pub struct PenroseReport {
    /// Number of tiles generated.
    pub tile_count: usize,
    /// Ratio of thick to thin tiles (should approach 1/φ ≈ 0.618).
    pub thick_thin_ratio: f64,
    /// Whether the ratio is within tolerance of 1/φ.
    pub ratio_ok: bool,
    /// Approximate 5-fold symmetry score (0..1).
    pub five_fold_score: f64,
    /// Whether 5-fold symmetry is above threshold.
    pub five_fold_ok: bool,
    /// Whether the pattern has no short-period repetition.
    pub aperiodic: bool,
    /// Minimum nearest-neighbour distance.
    pub min_nn_distance: f64,
    /// Overall pass/fail.
    pub passes: bool,
}

/// Generalized cut-and-project compiler.
pub struct CutAndProjectCompiler {
    pub(crate) source_dim: usize,
    pub(crate) target_dim: usize,
    /// Projection matrix: target_dim × source_dim.
    pub(crate) projection: Vec<Vec<f64>>,
    /// Perpendicular-space projection (computed from `projection`).
    pub(crate) perp_projection: Vec<Vec<f64>>,
    /// Acceptance window in perpendicular space.
    pub(crate) window_fn: Box<dyn Fn(&[f64]) -> bool>,
}

impl CutAndProjectCompiler {
    /// Create a new compiler with identity-ish projection (will be overridden).
    pub fn new(source_dim: usize, target_dim: usize) -> Self {
        assert!(
            target_dim <= source_dim,
            "target_dim must be ≤ source_dim"
        );
        // Start with zeros; caller should set a real projection.
        let projection = vec![vec![0.0; source_dim]; target_dim];
        let perp_dim = source_dim - target_dim;
        let perp_projection = vec![vec![0.0; source_dim]; perp_dim];
        Self {
            source_dim,
            target_dim,
            projection,
            perp_projection,
            window_fn: Box::new(|_| true),
        }
    }

    /// Standard Penrose P3 projection: 5D → 2D with golden-angle rotation.
    ///
    /// Each source axis is rotated by k · 2π/5 in the target plane.
    pub fn with_golden_projection(mut self) -> Self {
        assert!(
            self.source_dim == 5 && self.target_dim == 2,
            "Golden projection requires 5D → 2D"
        );
        let angle_step = 2.0 * std::f64::consts::PI / 5.0;
        for k in 0..5 {
            let angle = k as f64 * angle_step;
            self.projection[0][k] = angle.cos();
            self.projection[1][k] = angle.sin();
        }
        self.recompute_perp();
        // Default window: perpendicular-space coordinates within a strip of
        // half-width 1/φ centred on the origin.
        let hw = INV_PHI;
        self.window_fn = Box::new(move |perp: &[f64]| {
            perp.iter().all(|&v| v.abs() < hw)
        });
        self
    }

    /// Learned PCA projection from data.
    ///
    /// Performs a simple power-iteration PCA on the provided data to find
    /// the top-`target_dim` principal components, then uses those as the
    /// projection matrix.
    pub fn with_pca_projection(mut self, data: &[Vec<f64>]) -> Self {
        assert!(!data.is_empty(), "Need at least one data vector");
        assert!(
            data[0].len() == self.source_dim,
            "Data dimension must match source_dim"
        );

        let n = data.len();
        let d = self.source_dim;

        // Compute mean.
        let mut mean = vec![0.0; d];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }

        // Centre the data.
        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().enumerate().map(|(j, &v)| v - mean[j]).collect())
            .collect();

        // Covariance matrix (d × d), upper triangle.
        let mut cov = vec![vec![0.0; d]; d];
        for row in &centered {
            for i in 0..d {
                for j in i..d {
                    cov[i][j] += row[i] * row[j];
                }
            }
        }
        for i in 0..d {
            for j in i..d {
                cov[i][j] /= (n - 1).max(1) as f64;
                cov[j][i] = cov[i][j];
            }
        }

        // Power iteration for each target component.
        let iters = 200;
        for comp in 0..self.target_dim {
            let mut vec = vec![0.0; d];
            vec[comp % d] = 1.0;
            for _ in 0..iters {
                let mut new_vec = vec![0.0; d];
                for i in 0..d {
                    for j in 0..d {
                        new_vec[i] += cov[i][j] * vec[j];
                    }
                }
                // Deflate: subtract previously found components.
                for prev in 0..comp {
                    let dot: f64 = new_vec
                        .iter()
                        .zip(&self.projection[prev])
                        .map(|(a, b)| a * b)
                        .sum();
                    for k in 0..d {
                        new_vec[k] -= dot * self.projection[prev][k];
                    }
                }
                let norm = new_vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                for v in new_vec.iter_mut() {
                    *v /= norm;
                }
                vec = new_vec;
            }
            self.projection[comp] = vec.clone();
        }

        self.recompute_perp();

        // Data-driven window: use the spread of perpendicular-space projections.
        let perp_dim = self.source_dim - self.target_dim;
        let perp_maxes = if perp_dim > 0 && !centered.is_empty() {
            let mut maxes = vec![0.0f64; perp_dim];
            for row in &centered {
                let perp = self.project_perp(&row.iter().map(|&v| v).collect::<Vec<_>>());
                for (k, &v) in perp.iter().enumerate() {
                    maxes[k] = maxes[k].max(v.abs());
                }
            }
            maxes
        } else {
            vec![INV_PHI; perp_dim]
        };
        self.window_fn = Box::new(move |perp: &[f64]| {
            perp.iter().enumerate().all(|(k, &v)| v.abs() <= perp_maxes.get(k).copied().unwrap_or(INV_PHI) * 1.5)
        });
        self
    }

    /// Compile the tiling: scan a lattice cube, apply acceptance window,
    /// project accepted points.
    pub fn compile(&self, lattice_range: i32) -> Vec<TileCoord> {
        let mut tiles = Vec::new();
        self.scan_lattice(&mut tiles, lattice_range, 0, &mut vec![0i32; self.source_dim]);
        tiles
    }

    /// Verify that the compiled tiling has Penrose-like properties.
    pub fn verify_penrose(&self, tiles: &[TileCoord]) -> PenroseReport {
        let thick = tiles.iter().filter(|t| t.tile_type == TileType::Thick).count();
        let thin = tiles.iter().filter(|t| t.tile_type == TileType::Thin).count();
        let total = thick + thin;

        // thick:thin ratio — should approach 1/φ ≈ 0.618.
        let ratio = if thin > 0 {
            thick as f64 / thin as f64
        } else if thick > 0 {
            f64::INFINITY
        } else {
            0.0
        };
        let ratio_ok = total > 0 && (ratio - INV_PHI).abs() < 0.15;

        // 5-fold symmetry: rotate all tile coords by 72° and measure overlap.
        let five_fold_score = if !tiles.is_empty() && self.target_dim == 2 {
            self.compute_five_fold_score(tiles)
        } else {
            1.0
        };
        let five_fold_ok = five_fold_score > 0.3;

        // Aperiodicity: check no short-period translational repetition.
        let aperiodic = self.check_aperiodic(tiles);

        // Minimum nearest-neighbour distance.
        let min_nn = self.min_nearest_neighbour(tiles);

        let passes = ratio_ok && five_fold_ok && aperiodic && total > 0;
        PenroseReport {
            tile_count: total,
            thick_thin_ratio: ratio,
            ratio_ok,
            five_fold_score,
            five_fold_ok,
            aperiodic,
            min_nn_distance: min_nn,
            passes,
        }
    }

    // ── internal helpers ──────────────────────────────────────────

    /// Recompute the perpendicular-space projection via Gram-Schmidt on the
    /// complement of the row-space of `projection`.
    fn recompute_perp(&mut self) {
        let perp_dim = self.source_dim - self.target_dim;
        self.perp_projection = vec![vec![0.0; self.source_dim]; perp_dim];

        // Gram-Schmidt: start from standard basis, subtract projections onto
        // existing projection rows, collect remaining orthonormal vectors.
        let mut basis: Vec<Vec<f64>> = Vec::new();
        // First add the projection rows (normalised).
        for row in &self.projection {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                basis.push(row.iter().map(|&v| v / norm).collect());
            }
        }
        // Extend with standard basis vectors.
        for i in 0..self.source_dim {
            let mut e = vec![0.0; self.source_dim];
            e[i] = 1.0;
            // Subtract projections onto existing basis.
            for b in &basis {
                let dot: f64 = e.iter().zip(b).map(|(a, b)| a * b).sum();
                for k in 0..self.source_dim {
                    e[k] -= dot * b[k];
                }
            }
            let norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for v in e.iter_mut() {
                    *v /= norm;
                }
                if basis.len() < self.source_dim {
                    basis.push(e.clone());
                }
            }
        }

        // Fill perp_projection from the extra basis vectors.
        for (i, row) in basis[self.target_dim..].iter().enumerate() {
            if i < perp_dim {
                self.perp_projection[i] = row.clone();
            }
        }
    }

    /// Project source coordinates to target space.
    fn project(&self, source: &[i32]) -> Vec<f64> {
        let mut out = vec![0.0; self.target_dim];
        for (r, row) in self.projection.iter().enumerate() {
            for (c, &coeff) in row.iter().enumerate() {
                out[r] += coeff * source[c] as f64;
            }
        }
        out
    }

    /// Project source coordinates to perpendicular space.
    fn project_perp(&self, source: &[f64]) -> Vec<f64> {
        let perp_dim = self.perp_projection.len();
        let mut out = vec![0.0; perp_dim];
        for (r, row) in self.perp_projection.iter().enumerate() {
            for (c, &coeff) in row.iter().enumerate() {
                out[r] += coeff * source[c];
            }
        }
        out
    }

    /// Recursive lattice scan.
    fn scan_lattice(
        &self,
        tiles: &mut Vec<TileCoord>,
        range: i32,
        dim: usize,
        coords: &mut Vec<i32>,
    ) {
        if dim == self.source_dim {
            // Project to perpendicular space and check window.
            let source_f: Vec<f64> = coords.iter().map(|&v| v as f64).collect();
            let perp = self.project_perp(&source_f);
            if !(self.window_fn)(&perp) {
                return;
            }
            let target = self.project(coords);
            let tile_type = self.classify_tile(coords);
            tiles.push(TileCoord {
                x: target[0],
                y: if self.target_dim > 1 { target[1] } else { 0.0 },
                source_coords: coords.clone(),
                tile_type,
            });
        } else {
            for v in -range..=range {
                coords[dim] = v;
                self.scan_lattice(tiles, range, dim + 1, coords);
            }
        }
    }

    /// Classify a tile as thick or thin using the sum of coordinates mod golden ratio.
    fn classify_tile(&self, coords: &[i32]) -> TileType {
        let sum: f64 = coords.iter().map(|&v| (v as f64).abs()).sum::<f64>() * INV_PHI;
        let frac = sum - sum.floor();
        if frac < INV_PHI {
            TileType::Thick
        } else {
            TileType::Thin
        }
    }

    /// 5-fold symmetry score.
    fn compute_five_fold_score(&self, tiles: &[TileCoord]) -> f64 {
        let angle = 2.0 * std::f64::consts::PI / 5.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rotate each tile by 72° and check if a close neighbour exists.
        let mut matched = 0usize;
        let threshold = 0.5;

        for t in tiles.iter().take(500) {
            let rx = t.x * cos_a - t.y * sin_a;
            let ry = t.x * sin_a + t.y * cos_a;
            let has_match = tiles.iter().take(500).any(|u| {
                let dx = u.x - rx;
                let dy = u.y - ry;
                (dx * dx + dy * dy).sqrt() < threshold
            });
            if has_match {
                matched += 1;
            }
        }
        let n = tiles.iter().take(500).count();
        if n == 0 { 1.0 } else { matched as f64 / n as f64 }
    }

    /// Check that the tiling has no short-period repetition.
    fn check_aperiodic(&self, tiles: &[TileCoord]) -> bool {
        if tiles.len() < 10 {
            return true;
        }
        // Bin tiles along x-axis and check for periodicity in the bin counts.
        let n_bins = 50;
        let mut bins = vec![0u32; n_bins];
        if tiles.is_empty() {
            return true;
        }
        let x_min = tiles.iter().map(|t| t.x).fold(f64::INFINITY, f64::min);
        let x_max = tiles.iter().map(|t| t.x).fold(f64::NEG_INFINITY, f64::max);
        let span = (x_max - x_min).max(1e-12);
        for t in tiles {
            let idx = ((t.x - x_min) / span * (n_bins - 1) as f64).round() as usize;
            let idx = idx.min(n_bins - 1);
            bins[idx] += 1;
        }
        // Check periods 1..10.
        for period in 1..10 {
            let mut periodic = true;
            for i in period..n_bins {
                if bins[i] != bins[i - period] {
                    periodic = false;
                    break;
                }
            }
            if periodic {
                return false;
            }
        }
        true
    }

    /// Minimum nearest-neighbour distance.
    fn min_nearest_neighbour(&self, tiles: &[TileCoord]) -> f64 {
        if tiles.len() < 2 {
            return 0.0;
        }
        let mut min_d = f64::INFINITY;
        let limit = tiles.len().min(500);
        for i in 0..limit {
            for j in (i + 1)..limit {
                let dx = tiles[i].x - tiles[j].x;
                let dy = tiles[i].y - tiles[j].y;
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_d {
                    min_d = d;
                }
            }
        }
        if min_d == f64::INFINITY { 0.0 } else { min_d }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Golden projection produces tiles.
    #[test]
    fn test_golden_projection_produces_tiles() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(3);
        assert!(!tiles.is_empty(), "Golden projection should produce tiles");
    }

    // 2. No tiles when window rejects everything.
    #[test]
    fn test_reject_all_window() {
        let mut compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        compiler.window_fn = Box::new(|_| false);
        let tiles = compiler.compile(3);
        assert!(tiles.is_empty(), "Reject-all window should produce zero tiles");
    }

    // 3. Tile types are Thick or Thin (never Rejected in output).
    #[test]
    fn test_tile_types_valid() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(4);
        for t in &tiles {
            assert!(
                t.tile_type == TileType::Thick || t.tile_type == TileType::Thin,
                "Output tile should be Thick or Thin, got {:?}",
                t.tile_type
            );
        }
    }

    // 4. Source coords length matches source_dim.
    #[test]
    fn test_source_coords_dimension() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(2);
        for t in &tiles {
            assert_eq!(t.source_coords.len(), 5);
        }
    }

    // 5. Verify Penrose report runs and returns reasonable results.
    #[test]
    fn test_verify_penrose_report() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(5);
        let report = compiler.verify_penrose(&tiles);
        assert!(report.tile_count > 0);
        // The thick:thin ratio should be in a plausible range.
        assert!(report.thick_thin_ratio > 0.0);
    }

    // 6. Aperiodicity check on golden projection output.
    #[test]
    fn test_aperiodicity() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(6);
        let report = compiler.verify_penrose(&tiles);
        assert!(report.aperiodic, "Cut-and-project tiles should be aperiodic");
    }

    // 7. PCA projection from synthetic data.
    #[test]
    fn test_pca_projection() {
        // Generate synthetic 5D data with clear variance along first 2 axes.
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.cos(), t.sin(), 0.01 * t, 0.01 * t, 0.01 * t]
            })
            .collect();
        let compiler = CutAndProjectCompiler::new(5, 2).with_pca_projection(&data);
        let tiles = compiler.compile(3);
        // PCA should find the plane of variation and produce tiles.
        assert!(!tiles.is_empty(), "PCA projection should produce tiles from structured data");
    }

    // 8. Increasing lattice range produces more tiles.
    #[test]
    fn test_more_range_more_tiles() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let t1 = compiler.compile(2);
        let t2 = compiler.compile(4);
        assert!(
            t2.len() >= t1.len(),
            "Larger range should produce at least as many tiles: {} vs {}",
            t2.len(),
            t1.len()
        );
    }

    // 9. Compile with range=0 should produce the origin tile (if accepted).
    #[test]
    fn test_range_zero() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(0);
        // Origin may or may not pass the window; either way it should not panic.
        for t in &tiles {
            assert_eq!(t.source_coords.len(), 5);
        }
    }

    // 10. Projection matrix dimensions are correct.
    #[test]
    fn test_projection_dimensions() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        assert_eq!(compiler.projection.len(), 2);
        assert_eq!(compiler.projection[0].len(), 5);
        assert_eq!(compiler.perp_projection.len(), 3);
        assert_eq!(compiler.perp_projection[0].len(), 5);
    }

    // 11. Min nearest-neighbour distance is positive for non-trivial tilings.
    #[test]
    fn test_min_nn_positive() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(4);
        if tiles.len() > 1 {
            let report = compiler.verify_penrose(&tiles);
            assert!(
                report.min_nn_distance > 0.0,
                "Min NN distance should be positive, got {}",
                report.min_nn_distance
            );
        }
    }

    // 12. 5-fold symmetry on golden projection.
    #[test]
    fn test_five_fold_symmetry() {
        let compiler = CutAndProjectCompiler::new(5, 2).with_golden_projection();
        let tiles = compiler.compile(6);
        let report = compiler.verify_penrose(&tiles);
        assert!(
            report.five_fold_ok,
            "Golden projection should have 5-fold symmetry (score={:.3})",
            report.five_fold_score
        );
    }
}
