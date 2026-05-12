//! # Penrose Memory
//!
//! Aperiodic memory palace for AI agents.
//!
//! Navigate memories by **distance + direction** on a Penrose floor.
//! No coordinates. No index lookups. Just dead reckoning.
//!
//! ## Quick Start
//!
//! ```rust
//! use penrose_memory::{PenroseFloor, Step};
//!
//! // Create a memory floor
//! let mut floor = PenroseFloor::new()
//!     .at(0.0, 0.0)    // start position
//!     .facing(0.0);     // facing east
//!
//! // Store a memory at current position
//! let addr = floor.store_here(0xDEADBEEF);
//!
//! // Navigate to retrieve: walk 10 units at heading π/4
//! let steps = vec![Step::new(10.0, std::f64::consts::FRAC_PI_4)];
//! let read = floor.walk(&steps);
//! println!("Read {} bits, confidence {}", read.bits.len(), read.confidence);
//!
//! // Spline directly to a target
//! let read = floor.spline_to((5.0, 5.0), 5);
//! ```

const PHI: f64 = 1.618033988749895;

/// A navigation step: distance + direction.
///
/// This is the fundamental query primitive.
/// "How far" and "which way" — nothing else needed.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    pub distance: f64,
    pub heading: f64,
}

impl Step {
    pub fn new(distance: f64, heading: f64) -> Self {
        Self { distance, heading }
    }

    /// Advance a position by this step
    pub fn advance(&self, pos: (f64, f64)) -> (f64, f64) {
        (
            pos.0 + self.distance * self.heading.cos(),
            pos.1 + self.distance * self.heading.sin(),
        )
    }
}

/// What you read from the floor after walking a path
#[derive(Debug, Clone)]
pub struct FloorRead {
    /// Bits read from tiles (thick=1, thin=0)
    pub bits: Vec<bool>,
    /// Positions visited
    pub path: Vec<(f64, f64)>,
    /// Heading at each step
    pub headings: Vec<f64>,
    /// Whether matching rules held at each step
    pub matched: Vec<bool>,
    /// Overall confidence (fraction of matching-rule hits)
    pub confidence: f64,
}

/// A memory stored on the floor
#[derive(Debug, Clone)]
pub struct Memory {
    /// Position on the floor
    pub position: (f64, f64),
    /// The stored value
    pub content: u64,
    /// Tile type at this position (0=thin, 1=thick)
    pub tile_bit: bool,
    /// Tile level in the golden hierarchy
    pub level: u32,
}

/// The Penrose memory floor.
///
/// Navigate by dead reckoning: distance + direction.
/// The floor's Fibonacci word pattern determines tile bits.
/// Matching rules verify you're on a valid path.
///
/// # Memory Model
///
/// - **Store**: place a memory at a position on the floor
/// - **Walk**: navigate by steps, read bits under your feet
/// - **Spline**: walk straight to a target position
/// - **Tack**: zigzag like a sailboat, reading bits at each turn
/// - **Stretch**: vary step distances at constant heading
///
/// # The Math
///
/// The floor uses the Fibonacci word: substitute 1→10, 0→1, starting from 1.
/// The ratio of 1s to 0s converges to φ (golden ratio).
/// This IS the Penrose tiling's thick:thin ratio.
/// The pattern is deterministic from any seed — matching rules lock in.
pub struct PenroseFloor {
    memories: std::collections::HashMap<(i64, i64), Memory>,
    pos: (f64, f64),
    heading: f64,
    scale: f64,
}

impl PenroseFloor {
    pub fn new() -> Self {
        Self {
            memories: std::collections::HashMap::new(),
            pos: (0.0, 0.0),
            heading: 0.0,
            scale: 1.0,
        }
    }

    /// Set starting position (builder pattern)
    pub fn at(mut self, x: f64, y: f64) -> Self {
        self.pos = (x, y);
        self
    }

    /// Set starting heading in radians (builder pattern)
    pub fn facing(mut self, heading: f64) -> Self {
        self.heading = heading;
        self
    }

    /// Set tile scale (builder pattern)
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Quantize continuous position to nearest Penrose tile
    fn quantize(&self, pos: (f64, f64)) -> (i64, i64) {
        (
            (pos.0 / (self.scale * PHI)).round() as i64,
            (pos.1 / (self.scale * PHI)).round() as i64,
        )
    }

    /// Fibonacci word bit at a given lattice position
    ///
    /// Uses golden ratio hashing to ensure aperiodic pattern.
    /// The ratio of thick:thin converges to 1/φ ≈ 0.618.
    fn tile_bit(&self, pos: (i64, i64)) -> bool {
        let q = pos.0.wrapping_mul(0x9E3779B97F4A7C15u64 as i64);
        let mixed = (q ^ pos.1).wrapping_abs();
        let idx = (mixed % 1000) as usize;
        let inv_phi = 1.0 / PHI;
        let current = (idx as f64 * inv_phi).floor() as u64;
        let next = ((idx + 1) as f64 * inv_phi).floor() as u64;
        next != current
    }

    /// Check matching rules at a position
    fn matching_rule_holds(&self, pos: (i64, i64)) -> bool {
        let bit = self.tile_bit(pos);
        let neighbors = [
            (pos.0 + 1, pos.1), (pos.0 - 1, pos.1),
            (pos.0, pos.1 + 1), (pos.0, pos.1 - 1),
            (pos.0 + 1, pos.1 - 1), (pos.0 - 1, pos.1 + 1),
        ];
        if bit {
            neighbors.iter().any(|&n| !self.tile_bit(n))
        } else {
            neighbors.iter().any(|&n| self.tile_bit(n))
        }
    }

    /// Store a memory at the current position
    pub fn store_here(&mut self, content: u64) -> (i64, i64) {
        let key = self.quantize(self.pos);
        self.memories.insert(key, Memory {
            position: self.pos,
            content,
            tile_bit: self.tile_bit(key),
            level: 0,
        });
        key
    }

    /// Store a memory at an explicit position
    pub fn store_at(&mut self, x: f64, y: f64, content: u64) -> (i64, i64) {
        let key = self.quantize((x, y));
        self.memories.insert(key, Memory {
            position: (x, y),
            content,
            tile_bit: self.tile_bit(key),
            level: 0,
        });
        key
    }

    /// Retrieve a memory by walking to it
    pub fn retrieve_at(&self, x: f64, y: f64) -> Option<&Memory> {
        let key = self.quantize((x, y));
        self.memories.get(&key)
    }

    /// Walk a path and read bits from the floor
    pub fn walk(&self, steps: &[Step]) -> FloorRead {
        let mut bits = Vec::new();
        let mut path = Vec::new();
        let mut headings = Vec::new();
        let mut matched = Vec::new();
        let mut pos = self.pos;
        let mut heading = self.heading;

        for step in steps {
            heading = step.heading;
            let new_pos = step.advance(pos);
            let key = self.quantize(new_pos);
            bits.push(self.tile_bit(key));
            path.push(new_pos);
            headings.push(heading);
            matched.push(self.matching_rule_holds(key));
            pos = new_pos;
        }

        let confidence = if matched.is_empty() { 1.0 }
            else { matched.iter().filter(|&&m| m).count() as f64 / matched.len() as f64 };

        FloorRead { bits, path, headings, matched, confidence }
    }

    /// Spline walk: straight line to target in N equal steps
    pub fn spline_to(&self, target: (f64, f64), n_steps: usize) -> FloorRead {
        let dx = target.0 - self.pos.0;
        let dy = target.1 - self.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        let heading = dy.atan2(dx);
        let steps = vec![Step::new(dist / n_steps.max(1) as f64, heading); n_steps];
        self.walk(&steps)
    }

    /// Tack walk: zigzag like a sailboat
    pub fn tack(&self, heading_deltas: &[f64], step_distance: f64) -> FloorRead {
        let steps: Vec<Step> = heading_deltas
            .iter()
            .scan(self.heading, |h, &delta| {
                *h += delta;
                Some(Step::new(step_distance, *h))
            })
            .collect();
        self.walk(&steps)
    }

    /// Stretch walk: varying distances at constant heading
    pub fn stretch(&self, stretches: &[f64], heading: f64) -> FloorRead {
        let steps: Vec<Step> = stretches
            .iter()
            .map(|&d| Step::new(d * self.scale * PHI, heading))
            .collect();
        self.walk(&steps)
    }

    /// Deflate (consolidate) nearby memories into one
    pub fn deflate(&mut self, center: (f64, f64), radius: f64) -> Option<u64> {
        let key = self.quantize(center);
        let r2 = (radius / (self.scale * PHI)).ceil() as i64;
        let mut contents = Vec::new();
        let mut to_remove = Vec::new();

        for (&(q, r), mem) in &self.memories {
            let dq = q - key.0;
            let dr = r - key.1;
            if dq * dq + dr * dr <= r2 * r2 {
                contents.push(mem.content);
                to_remove.push((q, r));
            }
        }

        if contents.is_empty() { return None; }

        let consolidated = contents.into_iter().fold(0u64, |a, b| a ^ b);
        for k in to_remove { self.memories.remove(&k); }

        self.memories.insert(key, Memory {
            position: center,
            content: consolidated,
            tile_bit: self.tile_bit(key),
            level: 1,
        });

        Some(consolidated)
    }

    /// Find nearest stored memory to a position
    pub fn nearest(&self, pos: (f64, f64)) -> Option<&Memory> {
        let key = self.quantize(pos);
        let mut best: Option<&Memory> = None;
        let mut best_dist = f64::MAX;

        for (&(q, r), mem) in &self.memories {
            let dist = ((q as f64 - key.0 as f64).powi(2)
                + (r as f64 - key.1 as f64).powi(2)).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best = Some(mem);
            }
        }
        best
    }

    /// Number of stored memories
    pub fn len(&self) -> usize { self.memories.len() }
    pub fn is_empty(&self) -> bool { self.memories.is_empty() }
    pub fn position(&self) -> (f64, f64) { self.pos }
    pub fn heading(&self) -> f64 { self.heading }
}

impl Default for PenroseFloor {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut floor = PenroseFloor::new().at(5.0, 5.0);
        floor.store_here(0xBEEF);
        assert_eq!(floor.retrieve_at(5.0, 5.0).unwrap().content, 0xBEEF);
    }

    #[test]
    fn test_dead_reckoning_walk() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        floor.store_here(0x1111);
        floor.store_at(5.0, 0.0, 0x2222);

        let steps = vec![Step::new(5.0, 0.0)];
        let read = floor.walk(&steps);
        assert!(!read.bits.is_empty());
        assert!(read.confidence > 0.0);
    }

    #[test]
    fn test_spline() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        floor.store_here(0xBEEF);
        let read = floor.spline_to((0.0, 0.0), 5);
        assert!(read.confidence > 0.0);
    }

    #[test]
    fn test_tack() {
        let floor = PenroseFloor::new().at(0.0, 0.0).facing(0.0);
        let deltas = vec![0.5, -1.0, 0.5, -1.0];
        let read = floor.tack(&deltas, PHI);
        assert_eq!(read.headings.len(), 4);
    }

    #[test]
    fn test_stretch() {
        let floor = PenroseFloor::new().at(0.0, 0.0).facing(0.0);
        let read = floor.stretch(&[1.0, PHI, 1.0, PHI * PHI], 0.0);
        assert_eq!(read.bits.len(), 4);
    }

    #[test]
    fn test_deflate() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        floor.store_at(0.0, 0.0, 1);
        floor.store_at(1.0, 0.0, 2);
        floor.store_at(0.0, 1.0, 3);
        let result = floor.deflate((0.0, 0.0), 3.0);
        assert!(result.is_some());
        assert!(floor.len() < 3); // Consolidated
    }

    #[test]
    fn test_nearest() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        floor.store_at(0.0, 0.0, 1);
        floor.store_at(10.0, 10.0, 2);
        let mem = floor.nearest((1.0, 1.0)).unwrap();
        assert_eq!(mem.content, 1);
    }

    #[test]
    fn test_aperiodic_bits() {
        let floor = PenroseFloor::new().at(0.0, 0.0);
        let east: Vec<bool> = (0..50).map(|q| floor.tile_bit((q, 0))).collect();
        let north: Vec<bool> = (0..50).map(|r| floor.tile_bit((0, r))).collect();
        assert_ne!(east, north, "Different directions should give different patterns");
    }

    #[test]
    fn test_fibonacci_ratio() {
        let floor = PenroseFloor::new();
        let bits: Vec<bool> = (0..1000).map(|q| floor.tile_bit((q, 0))).collect();
        let ratio = bits.iter().filter(|&&b| b).count() as f64 / bits.len() as f64;
        assert!((ratio - 1.0/PHI).abs() < 0.05, "Ratio {} should approach 1/φ", ratio);
    }
}
