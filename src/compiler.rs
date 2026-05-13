//! High-level fleet tiling API.
//!
//! Takes real agent embedding vectors, fits a PCA projection, and compiles
//! an aperiodic tiling that assigns each agent to a unique tile.

use crate::cut_and_project::{CutAndProjectCompiler, TileCoord, TileType};

/// An agent's position in the fleet tiling.
#[derive(Debug, Clone)]
pub struct AgentTile {
    /// Agent index in the input embedding array.
    pub agent_index: usize,
    /// Assigned tile coordinates.
    pub tile: TileCoord,
    /// Distance from the agent's embedded position to the tile centre.
    pub assignment_distance: f64,
}

/// Complete fleet tiling result.
#[derive(Debug, Clone)]
pub struct FleetTiling {
    /// Source dimension of the embedding space.
    pub source_dim: usize,
    /// All compiled tiles.
    pub tiles: Vec<TileCoord>,
    /// Agent-to-tile assignments.
    pub assignments: Vec<AgentTile>,
    /// Number of thick tiles.
    pub thick_count: usize,
    /// Number of thin tiles.
    pub thin_count: usize,
    /// Thick:thin ratio.
    pub thick_thin_ratio: f64,
}

/// Compile a fleet tiling from agent embedding vectors.
///
/// 1. Fits a PCA projection from the data.
/// 2. Runs cut-and-project with the learned projection.
/// 3. Assigns each agent to the nearest compiled tile.
///
/// Returns a `FleetTiling` with tile assignments for each agent.
pub fn compile_fleet_tiling(agent_embeddings: &[Vec<f64>]) -> FleetTiling {
    assert!(
        !agent_embeddings.is_empty(),
        "Need at least one agent embedding"
    );

    let source_dim = agent_embeddings[0].len();
    let target_dim = 2;

    // Build compiler with PCA projection learned from agent embeddings.
    let compiler = CutAndProjectCompiler::new(source_dim, target_dim)
        .with_pca_projection(agent_embeddings);

    // Choose lattice range based on number of agents: enough tiles to cover.
    let lattice_range = ((agent_embeddings.len() as f64).sqrt().ceil() as i32).max(3);

    let tiles = compiler.compile(lattice_range);

    // Assign each agent to the nearest tile.
    let mut assignments = Vec::with_capacity(agent_embeddings.len());
    for (idx, emb) in agent_embeddings.iter().enumerate() {
        // Project the agent embedding to 2D using the same projection.
        let mut ax = 0.0f64;
        let mut ay = 0.0f64;
        let proj = &compiler.projection();
        for (r, row) in proj.iter().enumerate() {
            let mut val = 0.0f64;
            for (c, &coeff) in row.iter().enumerate() {
                if c < emb.len() {
                    val += coeff * emb[c];
                }
            }
            if r == 0 {
                ax = val;
            } else if r == 1 {
                ay = val;
            }
        }

        // Find nearest tile.
        let mut best_dist = f64::INFINITY;
        let mut best_tile_idx = 0usize;
        for (ti, tile) in tiles.iter().enumerate() {
            let dx = tile.x - ax;
            let dy = tile.y - ay;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_tile_idx = ti;
            }
        }

        assignments.push(AgentTile {
            agent_index: idx,
            tile: tiles[best_tile_idx].clone(),
            assignment_distance: best_dist,
        });
    }

    let thick_count = tiles.iter().filter(|t| t.tile_type == TileType::Thick).count();
    let thin_count = tiles.iter().filter(|t| t.tile_type == TileType::Thin).count();
    let thick_thin_ratio = if thin_count > 0 {
        thick_count as f64 / thin_count as f64
    } else if thick_count > 0 {
        f64::INFINITY
    } else {
        0.0
    };

    FleetTiling {
        source_dim,
        tiles,
        assignments,
        thick_count,
        thin_count,
        thick_thin_ratio,
    }
}

/// Public projection accessor on CutAndProjectCompiler.
impl CutAndProjectCompiler {
    /// Get a reference to the projection matrix.
    pub fn projection(&self) -> &Vec<Vec<f64>> {
        &self.projection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: generate synthetic agent embeddings in 5D.
    fn make_agents(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.5;
                vec![t.cos(), t.sin(), (t * 0.3).cos(), (t * 0.3).sin(), 0.01 * t]
            })
            .collect()
    }

    // 1. Basic fleet tiling compiles.
    #[test]
    fn test_fleet_tiling_compiles() {
        let agents = make_agents(10);
        let fleet = compile_fleet_tiling(&agents);
        assert!(!fleet.tiles.is_empty());
        assert_eq!(fleet.assignments.len(), 10);
    }

    // 2. Every agent gets an assignment.
    #[test]
    fn test_every_agent_assigned() {
        let agents = make_agents(20);
        let fleet = compile_fleet_tiling(&agents);
        for i in 0..20 {
            assert!(
                fleet.assignments.iter().any(|a| a.agent_index == i),
                "Agent {} should be assigned",
                i
            );
        }
    }

    // 3. Source dimension preserved.
    #[test]
    fn test_source_dim_preserved() {
        let agents = make_agents(5);
        let fleet = compile_fleet_tiling(&agents);
        assert_eq!(fleet.source_dim, 5);
    }

    // 4. Tile types are valid.
    #[test]
    fn test_fleet_tile_types_valid() {
        let agents = make_agents(10);
        let fleet = compile_fleet_tiling(&agents);
        for t in &fleet.tiles {
            assert!(t.tile_type == TileType::Thick || t.tile_type == TileType::Thin);
        }
    }

    // 5. Assignment distances are finite.
    #[test]
    fn test_assignment_distances_finite() {
        let agents = make_agents(15);
        let fleet = compile_fleet_tiling(&agents);
        for a in &fleet.assignments {
            assert!(
                a.assignment_distance.is_finite(),
                "Assignment distance should be finite"
            );
        }
    }

    // 6. Thick and thin counts sum to total.
    #[test]
    fn test_thick_thin_sum() {
        let agents = make_agents(10);
        let fleet = compile_fleet_tiling(&agents);
        assert_eq!(
            fleet.thick_count + fleet.thin_count,
            fleet.tiles.len(),
            "Thick + thin should equal total tiles"
        );
    }

    // 7. Single agent works.
    #[test]
    fn test_single_agent() {
        let agents = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]];
        let fleet = compile_fleet_tiling(&agents);
        assert_eq!(fleet.assignments.len(), 1);
        assert!(!fleet.tiles.is_empty());
    }

    // 8. Many agents (100).
    #[test]
    fn test_many_agents() {
        let agents = make_agents(100);
        let fleet = compile_fleet_tiling(&agents);
        assert_eq!(fleet.assignments.len(), 100);
        assert!(!fleet.tiles.is_empty());
    }
}
