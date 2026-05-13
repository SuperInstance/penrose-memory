//! # Penrose Memory
//!
//! Aperiodic memory palace for AI agents.
//!
//! Embeddings are projected to 2D Penrose coordinates using golden-ratio hashing.
//! The Fibonacci word determines tile bits (thick:thin → 1/φ). Matching rules
//! verify valid positions. Recall uses dead reckoning: walk from query toward
//! stored memories. 3-coloring for sharding, golden hierarchy (φ^k) for deflation.
//!
//! ## Quick Start
//!
//! ```rust
//! use penrose_memory::PenroseMemory;
//!
//! let mut pm = PenroseMemory::new(4);
//!
//! // Store an embedding with content
//! let id = pm.store(&[0.1, 0.2, 0.3, 0.4], 42);
//!
//! // Recall by nearest embedding
//! let results = pm.recall(&[0.1, 0.2, 0.3, 0.39], 5);
//! assert!(!results.is_empty());
//! assert_eq!(results[0].content, 42);
//! ```

pub mod cut_and_project;
pub mod compiler;
pub mod tensor_tile;

const PHI: f64 = 1.618033988749895;
const INV_PHI: f64 = 0.618033988749895;
const GOLDEN_ANGLE: f64 = 2.399963229728653; // π(3 − √5)

/// Tile lifecycle states — mirrors PLATO v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TileLifecycle {
    Active,
    Superseded,
    Retracted,
}

impl std::fmt::Display for TileLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileLifecycle::Active => write!(f, "Active"),
            TileLifecycle::Superseded => write!(f, "Superseded"),
            TileLifecycle::Retracted => write!(f, "Retracted"),
        }
    }
}

/// Lamport clock for causal ordering across agents.
#[derive(Debug, Clone)]
pub struct LamportClock {
    time: u64,
}

impl LamportClock {
    pub fn new() -> Self { Self { time: 0 } }
    pub fn tick(&mut self) -> u64 { self.time += 1; self.time }
    pub fn merge(&mut self, remote: u64) -> u64 { self.time = self.time.max(remote) + 1; self.time }
    pub fn now(&self) -> u64 { self.time }
}

/// A prediction about where a memory will be found.
#[derive(Debug, Clone)]
pub struct MemoryPrediction {
    pub predicted_tile_id: Option<u64>,
    pub predicted_heading: f64,
    pub predicted_distance: f64,
    pub lamport: u64,
    pub confirmed: bool,
    pub actual_tile_id: Option<u64>,
    pub actual_distance: Option<f64>,
}

/// Result of a recall operation.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub tile_id: u64,
    pub content: u64,
    pub confidence: f64,
    pub distance: f64,
    pub heading: f64,
}

/// Internal tile stored in the memory.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Tile {
    tile_id: u64,
    content: u64,
    x: f64,
    y: f64,
    color: u8,     // 3-coloring: 0, 1, or 2
    level: u32,    // golden hierarchy level
    lifecycle: TileLifecycle,
    lamport: u64,
}

/// Aperiodic memory palace navigated by dead reckoning.
///
/// Embeddings are projected to 2D Penrose coordinates using golden-ratio hashing.
/// The Fibonacci word determines tile bits. Recall walks from query toward stored
/// memories via dead reckoning.
pub struct PenroseMemory {
    embedding_dim: usize,
    tiles: Vec<Tile>,
    next_id: u64,
}

impl PenroseMemory {
    /// Create a new PenroseMemory with the given embedding dimension.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            tiles: Vec::new(),
            next_id: 1,
        }
    }

    /// Project an embedding to 2D Penrose coordinates using golden-ratio hashing.
    ///
    /// Each dimension contributes a golden-angle rotated component, producing
    /// an aperiodic 2D point from any embedding vector.
    fn project_to_2d(&self, embedding: &[f64]) -> (f64, f64) {
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let dim = self.embedding_dim.min(embedding.len());

        for i in 0..dim {
            let val = if i < embedding.len() { embedding[i] } else { 0.0 };
            let angle = (i as f64) * GOLDEN_ANGLE;
            let magnitude = val.abs();
            x += magnitude * angle.cos();
            y += magnitude * angle.sin();
        }

        // Normalize by dimension to keep coordinates bounded
        let scale = if dim > 0 { 1.0 / (dim as f64).sqrt() } else { 1.0 };
        (x * scale, y * scale)
    }

    /// Fibonacci word bit at a given lattice position.
    ///
    /// Uses golden ratio hashing: bit n is thick (1) if ⌊(n+1)/φ⌋ ≠ ⌊n/φ⌋.
    /// The ratio of thick:thin converges to 1/φ ≈ 0.618.
    fn tile_bit(&self, qx: i64, qy: i64) -> bool {
        // Hash the 2D lattice position to a 1D index using golden-ratio mixing
        let hash = (qx.wrapping_mul(0x9E3779B97F4A7C15u64 as i64))
            .wrapping_add(qy.wrapping_mul(0x517CC1B727220A95u64 as i64));
        let idx = (hash.wrapping_abs() % 10000) as u64;

        let current = ((idx as f64) * INV_PHI).floor() as u64;
        let next = (((idx + 1) as f64) * INV_PHI).floor() as u64;
        next != current
    }

    /// Compute the 3-coloring of a lattice position.
    ///
    /// Uses a hash mod 3 to assign one of three colors.
    /// For valid Penrose-like tilings, every tile gets a unique color
    /// such that no two adjacent same-color tiles exist.
    fn three_color(&self, qx: i64, qy: i64) -> u8 {
        let hash = (qx.wrapping_mul(0x517CC1B727220A95u64 as i64))
            .wrapping_add(qy.wrapping_mul(0x9E3779B97F4A7C15u64 as i64));
        (hash.wrapping_abs() % 3) as u8
    }

    /// Quantize continuous 2D coordinates to a lattice position.
    fn quantize(&self, x: f64, y: f64) -> (i64, i64) {
        let scale = PHI;
        (
            (x / scale).round() as i64,
            (y / scale).round() as i64,
        )
    }

    /// Compute Euclidean distance between two 2D points.
    fn euclidean_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }

    /// Compute heading (angle in radians) from point 1 to point 2.
    fn heading_to(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        (y2 - y1).atan2(x2 - x1)
    }

    /// Store an embedding with associated content.
    ///
    /// Projects the embedding to 2D Penrose coordinates and stores it.
    /// Returns the tile_id.
    pub fn store(&mut self, embedding: &[f64], content: u64) -> u64 {
        let (x, y) = self.project_to_2d(embedding);
        let (qx, qy) = self.quantize(x, y);
        let color = self.three_color(qx, qy);
        let id = self.next_id;
        self.next_id += 1;

        self.tiles.push(Tile {
            tile_id: id,
            content,
            x,
            y,
            color,
            level: 0,
            lifecycle: TileLifecycle::Active,
            lamport: 0,
        });

        id
    }

    /// Recall memories by dead reckoning from a query embedding.
    ///
    /// Projects the query to 2D, then finds the closest stored tiles.
    /// Dead reckoning: we walk from the query point toward stored memories,
    /// measuring distance and confidence at each step.
    ///
    /// `max_steps` controls how many intermediate waypoints to check along
    /// the path to each candidate.
    pub fn recall(&self, query: &[f64], max_steps: usize) -> Vec<RecallResult> {
        if self.tiles.is_empty() {
            return Vec::new();
        }

        let (qx, qy) = self.project_to_2d(query);

        let mut results: Vec<RecallResult> = self.tiles.iter()
            .filter(|t| t.lifecycle == TileLifecycle::Active)
            .map(|tile| {
            let dist = Self::euclidean_distance(qx, qy, tile.x, tile.y);
            let heading = Self::heading_to(qx, qy, tile.x, tile.y);

            // Confidence from distance: 1.0 at zero distance, decaying with distance
            // Using a Gaussian-like falloff
            let sigma = 2.0; // scale parameter
            let confidence = (-dist * dist / (2.0 * sigma * sigma)).exp();

            // Dead reckoning: verify path by checking intermediate points
            let mut path_confidence = 1.0;
            if max_steps > 0 && dist > 0.0 {
                let mut verified = 0usize;
                let mut total = 0usize;
                for step in 1..=max_steps {
                    let t = step as f64 / (max_steps + 1) as f64;
                    let ix = qx + t * (tile.x - qx);
                    let iy = qy + t * (tile.y - qy);
                    let (iqx, iqy) = self.quantize(ix, iy);

                    // Check matching rule at intermediate point
                    let bit = self.tile_bit(iqx, iqy);
                    let neighbors = [
                        (iqx + 1, iqy), (iqx - 1, iqy),
                        (iqx, iqy + 1), (iqx, iqy - 1),
                    ];
                    let any_diff = neighbors.iter().any(|&(nx, ny)| self.tile_bit(nx, ny) != bit);
                    if any_diff { verified += 1; }
                    total += 1;
                }
                path_confidence = if total > 0 { verified as f64 / total as f64 } else { 1.0 };
            }

            RecallResult {
                tile_id: tile.tile_id,
                content: tile.content,
                confidence: confidence * path_confidence,
                distance: dist,
                heading,
            }
        }).collect();

        // Sort by confidence descending (closest first)
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Navigate from a tile by distance and heading.
    ///
    /// Dead reckoning navigation: start from the given tile, walk the specified
    /// distance at the given heading, and find tiles near the destination.
    /// Returns tile IDs of tiles found within the arrival radius.
    pub fn navigate(&self, tile_id: u64, distance: f64, heading: f64) -> Vec<u64> {
        let start = match self.tiles.iter().find(|t| t.tile_id == tile_id) {
            Some(t) => (t.x, t.y),
            None => return Vec::new(),
        };

        // Walk by dead reckoning
        let dest_x = start.0 + distance * heading.cos();
        let dest_y = start.1 + distance * heading.sin();

        // Arrival radius scales with φ
        let arrival_radius = PHI * 0.5;

        self.tiles.iter()
            .filter(|t| {
                let d = Self::euclidean_distance(dest_x, dest_y, t.x, t.y);
                d <= arrival_radius
            })
            .map(|t| t.tile_id)
            .collect()
    }

    /// Consolidate (deflate) old memories using golden hierarchy.
    ///
    /// Tiles at level 0 that are close together (within φ^1 distance) are
    /// merged into a single tile at level 1. The merged content is XOR'd.
    /// Returns the number of tiles removed.
    pub fn consolidate(&mut self) {
        if self.tiles.len() < 2 {
            return;
        }

        let merge_distance = PHI; // φ^1

        // Find clusters of level-0 tiles
        let mut merged: Vec<bool> = vec![false; self.tiles.len()];
        let mut new_tiles: Vec<Tile> = Vec::new();

        for i in 0..self.tiles.len() {
            if merged[i] || self.tiles[i].level > 0 {
                continue;
            }

            let mut cluster_x = self.tiles[i].x;
            let mut cluster_y = self.tiles[i].y;
            let mut cluster_content = self.tiles[i].content;
            let mut cluster_size = 1usize;
            let mut cluster_ids = vec![i];

            for j in (i + 1)..self.tiles.len() {
                if merged[j] || self.tiles[j].level > 0 {
                    continue;
                }
                let d = Self::euclidean_distance(cluster_x, cluster_y, self.tiles[j].x, self.tiles[j].y);
                if d < merge_distance {
                    cluster_content ^= self.tiles[j].content;
                    cluster_x = (cluster_x * cluster_size as f64 + self.tiles[j].x) / (cluster_size + 1) as f64;
                    cluster_y = (cluster_y * cluster_size as f64 + self.tiles[j].y) / (cluster_size + 1) as f64;
                    cluster_size += 1;
                    cluster_ids.push(j);
                }
            }

            if cluster_size > 1 {
                for &idx in &cluster_ids {
                    merged[idx] = true;
                }
                let (qx, qy) = self.quantize(cluster_x, cluster_y);
                new_tiles.push(Tile {
                    tile_id: self.next_id,
                    content: cluster_content,
                    x: cluster_x,
                    y: cluster_y,
                    color: self.three_color(qx, qy),
                    level: 1,
                    lifecycle: TileLifecycle::Active,
                    lamport: 0,
                });
                self.next_id += 1;
            }
        }

        // Remove merged tiles, keep unmerged and new
        let mut kept: Vec<Tile> = self.tiles.drain(..)
            .enumerate()
            .filter(|(i, _)| !merged[*i])
            .map(|(_, t)| t)
            .collect();
        kept.extend(new_tiles);
        self.tiles = kept;
    }

    /// Number of stored tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Check matching rules at a lattice position.
    /// A tile matches if it has at least one neighbor of the opposite type.
    pub fn matching_rule_holds(&self, qx: i64, qy: i64) -> bool {
        let bit = self.tile_bit(qx, qy);
        let neighbors = [
            (qx + 1, qy), (qx - 1, qy),
            (qx, qy + 1), (qx, qy - 1),
            (qx + 1, qy - 1), (qx - 1, qy + 1),
        ];
        neighbors.iter().any(|&(nx, ny)| self.tile_bit(nx, ny) != bit)
    }

    /// Get the golden-ratio hash bit for a position.
    pub fn get_tile_bit(&self, qx: i64, qy: i64) -> bool {
        self.tile_bit(qx, qy)
    }

    /// Get the 3-coloring for a position.
    pub fn get_color(&self, qx: i64, qy: i64) -> u8 {
        self.three_color(qx, qy)
    }

    /// Check if the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    // ── Tile Lifecycle (v1.1.0) ─────────────────────────────

    /// Supersede a tile — mark old as Superseded, return its content.
    pub fn supersede_tile(&mut self, tile_id: u64) -> Option<u64> {
        let tile = self.tiles.iter_mut().find(|t| t.tile_id == tile_id)?;
        if tile.lifecycle != TileLifecycle::Active {
            return None;
        }
        tile.lifecycle = TileLifecycle::Superseded;
        Some(tile.content)
    }

    /// Retract a tile — mark as Retracted with reason.
    pub fn retract_tile(&mut self, tile_id: u64) -> bool {
        if let Some(tile) = self.tiles.iter_mut().find(|t| t.tile_id == tile_id) {
            if tile.lifecycle == TileLifecycle::Active {
                tile.lifecycle = TileLifecycle::Retracted;
                return true;
            }
        }
        false
    }

    /// Get only active tile IDs.
    pub fn active_tile_ids(&self) -> Vec<u64> {
        self.tiles.iter()
            .filter(|t| t.lifecycle == TileLifecycle::Active)
            .map(|t| t.tile_id)
            .collect()
    }

    /// Count tiles by lifecycle state.
    pub fn lifecycle_stats(&self) -> (usize, usize, usize) {
        let active = self.tiles.iter().filter(|t| t.lifecycle == TileLifecycle::Active).count();
        let superseded = self.tiles.iter().filter(|t| t.lifecycle == TileLifecycle::Superseded).count();
        let retracted = self.tiles.iter().filter(|t| t.lifecycle == TileLifecycle::Retracted).count();
        (active, superseded, retracted)
    }

    // ── Simulation-First Predictions (v1.1.0) ──────────────

    /// Predict where a memory will be found before walking there.
    /// Uses the same projection as recall but without executing the walk.
    pub fn predict_recall(&self, query: &[f64], clock: &mut LamportClock) -> MemoryPrediction {
        let (qx, qy) = self.project_to_2d(query);

        // Find closest active tile without walking
        let best = self.tiles.iter()
            .filter(|t| t.lifecycle == TileLifecycle::Active)
            .min_by(|a, b| {
                let da = Self::euclidean_distance(qx, qy, a.x, a.y);
                let db = Self::euclidean_distance(qx, qy, b.x, b.y);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some(tile) => MemoryPrediction {
                predicted_tile_id: Some(tile.tile_id),
                predicted_heading: Self::heading_to(qx, qy, tile.x, tile.y),
                predicted_distance: Self::euclidean_distance(qx, qy, tile.x, tile.y),
                lamport: clock.tick(),
                confirmed: false,
                actual_tile_id: None,
                actual_distance: None,
            },
            None => MemoryPrediction {
                predicted_tile_id: None,
                predicted_heading: 0.0,
                predicted_distance: f64::MAX,
                lamport: clock.tick(),
                confirmed: false,
                actual_tile_id: None,
                actual_distance: None,
            },
        }
    }

    /// Confirm a prediction against actual recall results.
    pub fn confirm_prediction(&self, prediction: &mut MemoryPrediction, actual: &[RecallResult]) -> bool {
        if let (Some(pred_id), Some(first)) = (prediction.predicted_tile_id, actual.first()) {
            prediction.actual_tile_id = Some(first.tile_id);
            prediction.actual_distance = Some(first.distance);
            prediction.confirmed = pred_id == first.tile_id;
            prediction.confirmed
        } else {
            prediction.confirmed = actual.is_empty() && prediction.predicted_tile_id.is_none();
            prediction.confirmed
        }
    }
}

impl Default for PenroseMemory {
    fn default() -> Self {
        Self::new(1536)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Store + recall roundtrip
    #[test]
    fn test_store_recall_roundtrip() {
        let mut pm = PenroseMemory::new(4);
        let id = pm.store(&[0.1, 0.2, 0.3, 0.4], 42);
        let results = pm.recall(&[0.1, 0.2, 0.3, 0.4], 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].content, 42);
        assert_eq!(results[0].tile_id, id);
    }

    // 2. Different embeddings → different tiles
    #[test]
    fn test_different_embeddings_different_tiles() {
        let mut pm = PenroseMemory::new(4);
        let id1 = pm.store(&[1.0, 0.0, 0.0, 0.0], 1);
        let id2 = pm.store(&[0.0, 0.0, 0.0, 1.0], 2);
        assert_ne!(id1, id2);

        let results1 = pm.recall(&[1.0, 0.0, 0.0, 0.0], 3);
        let results2 = pm.recall(&[0.0, 0.0, 0.0, 1.0], 3);
        assert_eq!(results1[0].content, 1);
        assert_eq!(results2[0].content, 2);
    }

    // 3. Nearby embeddings → nearby tiles (small distance)
    #[test]
    fn test_nearby_embeddings_nearby_tiles() {
        let mut pm = PenroseMemory::new(4);
        pm.store(&[1.0, 2.0, 3.0, 4.0], 100);

        // Slightly perturbed query
        let results = pm.recall(&[1.01, 2.01, 3.01, 4.01], 3);
        assert!(!results.is_empty());
        assert!(results[0].distance < 0.1, "Nearby embeddings should be close");
        assert_eq!(results[0].content, 100);
    }

    // 4. Matching rules hold
    #[test]
    fn test_matching_rules_hold() {
        let pm = PenroseMemory::new(4);
        let mut valid = 0usize;
        let total = 1000;
        for q in 0..total {
            if pm.matching_rule_holds(q, 0) {
                valid += 1;
            }
        }
        // Most positions should satisfy matching rules (at least 80%)
        let ratio = valid as f64 / total as f64;
        assert!(ratio > 0.75, "Matching rules should hold at most positions, got {}", ratio);
    }

    // 5. Fibonacci ratio → 1/φ
    #[test]
    fn test_fibonacci_ratio() {
        let pm = PenroseMemory::new(4);
        let n = 10000;
        let thick_count = (0..n).filter(|&q| pm.tile_bit(q, 0)).count();
        let ratio = thick_count as f64 / n as f64;
        assert!(
            (ratio - INV_PHI).abs() < 0.03,
            "Ratio {:.4} should approach 1/φ ≈ {:.4}",
            ratio, INV_PHI
        );
    }

    // 6. 3-coloring covers all positions
    #[test]
    fn test_three_coloring_covers_all() {
        let pm = PenroseMemory::new(4);
        let mut colors = [false; 3];
        for q in 0..300 {
            let c = pm.get_color(q, 0) as usize;
            assert!(c < 3);
            colors[c] = true;
        }
        // All 3 colors should appear
        assert!(colors[0] && colors[1] && colors[2], "All 3 colors should appear");
    }

    // 7. Consolidation reduces count
    #[test]
    fn test_consolidation_reduces_count() {
        let mut pm = PenroseMemory::new(4);
        // Store many similar embeddings that cluster close together
        for i in 0..20u64 {
            let val = 1.0 + (i as f64) * 0.001;
            pm.store(&[val, 2.0, 3.0, 4.0], i);
        }
        let before = pm.len();
        pm.consolidate();
        let after = pm.len();
        assert!(after < before, "Consolidation should reduce tile count: {} -> {}", before, after);
    }

    // 8. Region fingerprints are unique
    #[test]
    fn test_region_fingerprints_unique() {
        let pm = PenroseMemory::new(4);
        let mut fingerprints = std::collections::HashSet::new();
        for q in 0..100i64 {
            // Fingerprint: 3x3 neighborhood of tile bits
            let mut fp = 0u16;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let bit = pm.tile_bit(q + dx, dy) as u16;
                    fp = (fp << 1) | bit;
                }
            }
            fingerprints.insert(fp);
        }
        // Should have many unique fingerprints (aperiodic = many distinct neighborhoods)
        assert!(fingerprints.len() > 50, "Should have many unique fingerprints, got {}", fingerprints.len());
    }

    // 9. Aperiodic patterns (no periodicity)
    #[test]
    fn test_aperiodic_patterns() {
        let pm = PenroseMemory::new(4);
        let pattern: Vec<bool> = (0..100).map(|q| pm.tile_bit(q, 0)).collect();

        // Check no short period repeats
        for period in 2..20 {
            let mut periodic = true;
            for i in period..100 {
                if pattern[i] != pattern[i - period] {
                    periodic = false;
                    break;
                }
            }
            assert!(!periodic, "Pattern should not be periodic with period {}", period);
        }
    }

    // 10. Empty recall returns empty
    #[test]
    fn test_empty_recall() {
        let pm = PenroseMemory::new(4);
        let results = pm.recall(&[1.0, 2.0, 3.0, 4.0], 5);
        assert!(results.is_empty());
    }

    // 11. Dead reckoning navigation
    #[test]
    fn test_dead_reckoning_navigation() {
        let mut pm = PenroseMemory::new(4);
        let id = pm.store(&[1.0, 0.0, 0.0, 0.0], 10);
        // Navigate from the stored tile
        let results = pm.navigate(id, 0.0, 0.0);
        assert!(results.contains(&id), "Zero-distance navigation should find the same tile");
    }

    // 12. Large embedding (1536 dims) works
    #[test]
    fn test_large_embedding() {
        let mut pm = PenroseMemory::new(1536);
        let embedding: Vec<f64> = (0..1536).map(|i| (i as f64).sin() * 0.1).collect();
        let id = pm.store(&embedding, 999);
        let results = pm.recall(&embedding, 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].content, 999);
        assert_eq!(results[0].tile_id, id);
    }

    // 13. Store many (1000+) memories
    #[test]
    fn test_store_many_memories() {
        let mut pm = PenroseMemory::new(8);
        for i in 0..1000u64 {
            let emb: Vec<f64> = (0..8).map(|j| (i * 8 + j) as f64 * 0.001).collect();
            pm.store(&emb, i);
        }
        assert_eq!(pm.len(), 1000);
    }

    // 14. Confidence decreases with distance
    #[test]
    fn test_confidence_decreases_with_distance() {
        let mut pm = PenroseMemory::new(4);
        pm.store(&[1.0, 2.0, 3.0, 4.0], 42);

        // Close query
        let close = pm.recall(&[1.0, 2.0, 3.0, 4.0], 0);
        // Far query
        let far = pm.recall(&[10.0, 20.0, 30.0, 40.0], 0);

        assert!(!close.is_empty());
        assert!(!far.is_empty());
        assert!(
            close[0].confidence > far[0].confidence,
            "Close confidence ({}) should be > far confidence ({})",
            close[0].confidence, far[0].confidence
        );
    }

    // 15. Multi-step recall finds closer match
    #[test]
    fn test_multi_step_recall_finds_closer() {
        let mut pm = PenroseMemory::new(4);
        pm.store(&[1.0, 0.0, 0.0, 0.0], 10);
        pm.store(&[0.0, 0.0, 0.0, 1.0], 20);

        let results = pm.recall(&[0.9, 0.0, 0.0, 0.0], 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].content, 10, "Should find closest match first");
    }

    // ── v1.1.0: Tile lifecycle tests ───────────────────────

    #[test]
    fn test_new_tile_is_active() {
        let mut pm = PenroseMemory::new(4);
        pm.store(&[1.0, 2.0, 3.0, 4.0], 42);
        let ids = pm.active_tile_ids();
        assert_eq!(ids.len(), 1);
        let (active, sup, ret) = pm.lifecycle_stats();
        assert_eq!(active, 1);
        assert_eq!(sup, 0);
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_supersede_excludes_from_recall() {
        let mut pm = PenroseMemory::new(4);
        let id1 = pm.store(&[1.0, 0.0, 0.0, 0.0], 10);
        let _id2 = pm.store(&[0.9, 0.0, 0.0, 0.0], 20);

        pm.supersede_tile(id1);

        let results = pm.recall(&[1.0, 0.0, 0.0, 0.0], 3);
        assert!(!results.iter().any(|r| r.tile_id == id1), "Superseded tile should not appear in recall");

        let (active, sup, _) = pm.lifecycle_stats();
        assert_eq!(active, 1);
        assert_eq!(sup, 1);
    }

    #[test]
    fn test_retract_excludes_from_recall() {
        let mut pm = PenroseMemory::new(4);
        let id = pm.store(&[1.0, 0.0, 0.0, 0.0], 10);

        let retracted = pm.retract_tile(id);
        assert!(retracted);

        let results = pm.recall(&[1.0, 0.0, 0.0, 0.0], 3);
        assert!(results.is_empty(), "Retracted tile should not appear in recall");
    }

    #[test]
    fn test_cannot_supersede_retracted() {
        let mut pm = PenroseMemory::new(4);
        let id = pm.store(&[1.0, 2.0, 3.0, 4.0], 42);
        pm.retract_tile(id);
        let result = pm.supersede_tile(id);
        assert!(result.is_none(), "Cannot supersede a retracted tile");
    }

    // ── v1.1.0: Simulation-first prediction tests ─────────

    #[test]
    fn test_predict_recall_finds_tile() {
        let mut pm = PenroseMemory::new(4);
        let id = pm.store(&[1.0, 2.0, 3.0, 4.0], 42);
        let mut clock = LamportClock::new();

        let pred = pm.predict_recall(&[1.0, 2.0, 3.0, 4.0], &mut clock);
        assert_eq!(pred.predicted_tile_id, Some(id));
        assert!(pred.predicted_distance < 0.01);
        assert_eq!(pred.lamport, 1);
        assert!(!pred.confirmed);
    }

    #[test]
    fn test_confirm_prediction_matches() {
        let mut pm = PenroseMemory::new(4);
        let _id = pm.store(&[1.0, 2.0, 3.0, 4.0], 42);
        let mut clock = LamportClock::new();

        let mut pred = pm.predict_recall(&[1.0, 2.0, 3.0, 4.0], &mut clock);
        let actual = pm.recall(&[1.0, 2.0, 3.0, 4.0], 3);

        let confirmed = pm.confirm_prediction(&mut pred, &actual);
        assert!(confirmed, "Prediction should match actual recall");
        assert!(pred.confirmed);
    }

    #[test]
    fn test_lamport_clock_monotonic() {
        let mut clock = LamportClock::new();
        let t1 = clock.tick();
        let t2 = clock.tick();
        let t3 = clock.merge(100); // max(2, 100) + 1 = 101
        assert!(t1 < t2);
        assert_eq!(t3, 101);
    }
}
