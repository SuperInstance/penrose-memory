# Penrose Memory

Aperiodic memory palace for AI agents. Navigate memories by **distance + direction** on a Penrose floor.

## How It Works

1. **Project**: Embeddings → 2D Penrose coordinates via golden-ratio hashing
2. **Store**: Place memories on the floor at projected coordinates
3. **Recall**: Dead-reckon from query toward stored memories
4. **Navigate**: Walk from any tile by distance + heading
5. **Consolidate**: Merge nearby memories using golden hierarchy (φ^k)

The Fibonacci word determines tile bits (thick:thin → 1/φ). Matching rules verify valid positions. 3-coloring enables sharding.

## Install

```toml
[dependencies]
penrose-memory = "1.0.0"
```

Python:
```bash
pip install penrose-memory
```

## Quick Start (Rust)

```rust
use penrose_memory::PenroseMemory;

let mut pm = PenroseMemory::new(1536);

// Store an embedding with content
let id = pm.store(&embedding, 42);

// Recall by nearest embedding
let results = pm.recall(&query, 5);
for r in &results {
    println!("id={} conf={:.3} dist={:.3}", r.tile_id, r.confidence, r.distance);
}

// Navigate from a tile
let nearby = pm.navigate(id, 2.5, std::f64::consts::FRAC_PI_4);

// Consolidate old memories
pm.consolidate();
```

## Quick Start (Python)

```python
from penrose_memory import PenroseMemory

pm = PenroseMemory(embedding_dim=1536)

# Store text with embedding
tile_id = pm.store("hello world", embedding)

# Recall by query embedding
results = pm.recall(query_embedding, max_steps=5)
for r in results:
    print(r["text"], r["confidence"], r["distance"])

# Navigate from a tile
nearby = pm.navigate(tile_id, distance=2.5, heading=0.785)

# Consolidate
removed = pm.consolidate()
```

## API

### Rust

| Method | Description |
|--------|-------------|
| `new(dim)` | Create with embedding dimension |
| `store(&[f64], u64) -> u64` | Store embedding + content, returns tile_id |
| `recall(&[f64], steps) -> Vec<RecallResult>` | Recall by dead reckoning |
| `navigate(id, dist, heading) -> Vec<u64>` | Navigate from tile |
| `consolidate()` | Merge nearby memories |
| `len() -> usize` | Memory count |

### Python

| Method | Description |
|--------|-------------|
| `__init__(embedding_dim=1536)` | Create with embedding dimension |
| `store(text, embedding) -> int` | Store text + embedding |
| `recall(query_embedding, max_steps=5) -> list` | Recall by dead reckoning |
| `navigate(tile_id, distance, heading) -> list` | Navigate from tile |
| `consolidate() -> int` | Merge nearby memories |
| `__len__()` | Memory count |

## Tests

**Rust**: 15 tests covering roundtrip, aperiodicity, Fibonacci ratio, 3-coloring, consolidation, navigation, large embeddings, confidence decay, and more.

**Python**: 10 tests covering the same core functionality.

## The Penrose Tiling

A Penrose tiling is an aperiodic tiling — it covers the plane without ever repeating. The tiles come in two shapes (kite and dart, or thick and thin rhombi) arranged according to matching rules that forbid periodic patterns. This non-repeating structure means every location in the memory palace is unique and addressable by both distance and direction, unlike a regular grid where many positions look identical.

The golden ratio (φ ≈ 1.618) governs the tiling: the ratio of thick to thin tiles is φ, the Fibonacci word determines tile placement, and the consolidation hierarchy uses powers of φ as distance thresholds. This gives Penrose Memory a natural multi-scale structure — memories cluster at every scale without ever forming a rigid lattice.

## Related Repos

- **[plato-core](https://github.com/SuperInstance/plato-core)** — Foundation types and mesh registry
- **[cocapn-plato](https://github.com/SuperInstance/cocapn-plato)** — PLATO integration for knowledge rooms
- **[constraint-instrument](https://github.com/SuperInstance/constraint-instrument)** — Constraint-based music generation
- **[tensor-spline](https://github.com/SuperInstance/tensor-spline)** — Compressed neural network layers

## License

MIT
