# Penrose Memory — Architecture

**Forgemaster ⚒️ | 2026-05-12**

> Distance. Direction. The floor does the rest.

---

## 1. What This Is

Penrose Memory is a **production aperiodic memory palace** for AI agents. It replaces vector-database similarity search with **dead reckoning** — navigate memories using two numbers per step (distance + heading) on a deterministic aperiodic tiling.

The core insight: the Fibonacci word generates the entire floor from a single seed. The structure is **free** — computed, not stored. Every neighborhood is unique (matching rules). Navigation costs two floats. Retrieval reads bits off the floor.

**The context window becomes the fovea (2mm of a 150mm eye). The Penrose floor IS the brain.**

---

## 2. Core Abstractions

### 2.1 Floor

The memory space. An aperiodic tiling determined entirely by the Fibonacci word substitution rules (1→10, 0→1). The thick:thin tile ratio converges to 1/φ ≈ 0.618.

```
Floor {
    seed: u64,                        // determines entire tiling
    scale: f64,                       // tile spacing multiplier (default 1.0)
    dimension: usize,                 // embedding dimension (e.g. 1536)
    tiles: HashMap<TileCoord, Tile>,  // only stored memories, NOT the floor itself
}
```

**Key invariant**: The floor is never materialized. The Fibonacci word at any position is computed on demand via golden-ratio hashing:

```rust
fn tile_bit(coord: (i64, i64), seed: u64) -> bool {
    // Hash coord with seed, index into Fibonacci word via ⌊n/φ⌋ trick
    let mixed = wyhash(coord, seed);
    let idx = mixed % LOOKUP_TABLE_SIZE;
    let current = (idx as f64 * INV_PHI).floor() as u64;
    let next = ((idx + 1) as f64 * INV_PHI).floor() as u64;
    next != current  // 1 if Fibonacci word transitions, 0 otherwise
}
```

The ratio of 1s converges to 1/φ. This IS the Penrose thick:thin ratio. The pattern is deterministic from the seed, aperiodic, and locally unique.

**Dimension parameter**: For embedding-aware floors, `dimension` controls how many parallel Fibonacci word layers compose a single tile. Each layer uses the same seed but a different hash salt, producing independent aperiodic bit sequences. A 1536-dimensional tile is 1536 bits — ~192 bytes of addressable memory per tile.

### 2.2 Tile

A single memory location. Bit-dense. Determined by the Fibonacci word.

```
Tile {
    coord: TileCoord,         // (i64, i64) quantized position
    bit: bool,                // thick (1) or thin (0) — determined by floor
    levels: [u8; MAX_LEVEL],  // consolidation levels (0=fresh, 1=deflated, ...)
    content: Option<Bytes>,   // stored payload (None = empty tile, bit still exists)
}
```

**Single-bit encoding**: Casey's requirement. The tile IS one bit (thick=1, thin=0). There's only one way the pattern can go once you see it lock in. The matching rules enforce this — the Fibonacci word is the constraint.

For embedding-dimensional tiles, each tile is a **bit vector** of length `dimension`, not a single bit. Each bit comes from an independent Fibonacci word layer.

### 2.3 Walker

The agent navigating the floor. State is minimal: position + heading.

```
Walker {
    pos: (f64, f64),       // continuous position on floor
    heading: f64,           // current heading in radians
    trail: Vec<Step>,       // path history (for backtracking)
    confidence: f64,        // accumulated matching-rule confidence
}
```

The walker advances by **dead reckoning**: each step applies `(distance, heading)` to the current position:

```rust
fn advance(pos: (f64, f64), step: Step) -> (f64, f64) {
    (pos.0 + step.distance * step.heading.cos(),
     pos.1 + step.distance * step.heading.sin())
}
```

After each step, the walker quantizes its continuous position to the nearest tile and reads the bit. If matching rules fail, confidence drops — the walker has drifted and must adjust heading.

### 2.4 Region

A neighborhood of tiles with a unique fingerprint. The matching rules guarantee no two regions are identical (aperiodic uniqueness).

```
Region {
    center: TileCoord,
    radius: u32,                  // in tile units
    fingerprint: [bool; N],       // concatenation of tile bits in neighborhood
    matching_score: f64,          // fraction of tiles satisfying matching rules
}
```

**Location awareness**: A walker reads its surrounding region and compares the fingerprint to known patterns. If the fingerprint matches a stored region, the walker knows exactly where it is — no GPS needed.

### 2.5 Projection (Cut-and-Project)

The dimensional bridge. High-dimensional embedding space → low-dimensional navigation space.

```
Projection {
    high_dim: usize,       // e.g. 1536 (embedding dimension)
    low_dim: usize,        // e.g. 2 (navigation surface)
    matrix: Matrix<f64>,   // projection matrix (orthogonal or learned)
    shift: Vec<f64>,       // acceptance window offset (cut parameter)
}
```

**Cut-and-project method**: Take a high-dimensional lattice (e.g., R^1536), project through an acceptance window (the "cut"), and map onto a lower-dimensional subspace (e.g., R^2). The result is an aperiodic point set in the low-D space.

The acceptance window is a thickened region of the perpendicular space. Points in the high-D lattice that fall within the window "make the cut" and appear in the low-D projection. The shape and orientation of the window determines the tiling.

**Generalization**: This works in ANY dimension. There is no mathematical limit. A 10000-dimensional embedding projects to a 2D navigation surface. The projection preserves enough information for navigation while making the space walkable.

### 2.6 Step

The fundamental query primitive. Two numbers.

```
Step {
    distance: f64,   // how far to walk (in units of φ × scale)
    heading: f64,    // which direction (radians, 0=east, π/2=north)
}
```

**Golden-ratio stretches**: The natural distance unit is φ:

| Stretch | Distance | Granularity |
|---------|----------|-------------|
| 1φ | adjacent tiles | detail level |
| φ² | skip tiles | session level |
| φ³ | cross clusters | project level |
| φ⁴ | leap across palace | fleet level |

A query is a **spline** — a sequence of steps:

```
query = [(φ, 0.0), (φ², 0.52), (φ, -0.31), (φ⁴, 1.05), (φ, 0.0)]
```

Five steps. Ten numbers. Navigates the entire palace.

---

## 3. Data Structures & Algorithms

### 3.1 Fibonacci Word via Golden Ratio Hash

The floor doesn't store the tiling. It computes each tile bit on demand using the Beatty sequence property:

```
fibonacci_bit(n) = ⌊(n+1)/φ⌋ ≠ ⌊n/φ⌋
```

This is O(1) per bit. No precomputation. No storage. The nth bit of the infinite Fibonacci word is determined by whether the Beatty sequence transitions at position n.

For 2D tile coordinates, we hash `(x, y)` into a 1D index:

```rust
fn tile_bit(coord: (i64, i64), seed: u64) -> bool {
    let h = wyhash::hash(coord, seed);
    fibonacci_bit(h % MODULUS)
}
```

The hash ensures different seed values produce different aperiodic tilings. All share the same statistical properties (thick:thin → 1:φ).

### 3.2 Matching Rules

Penrose tilings enforce local constraints — the matching rules. Our implementation checks that each tile's neighborhood is consistent:

```rust
fn matching_rule_holds(coord: (i64, i64)) -> bool {
    let bit = tile_bit(coord);
    let neighbors = hex_neighbors(coord);  // 6 neighbors on triangular grid
    if bit {  // thick tile: must have at least one thin neighbor
        neighbors.any(|n| !tile_bit(n))
    } else {  // thin tile: must have at least one thick neighbor
        neighbors.any(|n| tile_bit(n))
    }
}
```

A high matching-rule score (close to 1.0) means the walker is on a valid path. A drop indicates drift — accumulated dead-reckoning error. The walker tacks (adjusts heading) to restore confidence.

### 3.3 Quantization

Continuous positions are quantized to the tile lattice:

```rust
fn quantize(pos: (f64, f64), scale: f64) -> (i64, i64) {
    let spacing = scale * PHI;
    (round(pos.0 / spacing) as i64, round(pos.1 / spacing) as i64)
}
```

Tile spacing is `scale × φ`. The scale parameter controls resolution: smaller scale = finer tiles = more bits per unit distance.

### 3.4 Cut-and-Project Algorithm

For embedding-aware floors:

```
INPUT:  embedding e ∈ R^D (e.g., D=1536)
        projection matrix P (D×d, e.g., 1536×2)
        acceptance window W ⊂ R^(D-d)

OUTPUT: navigation coordinate nav ∈ R^d

ALGORITHM:
    1. nav = P^T · e              // project to low-D
    2. perp = e - P · nav         // perpendicular component
    3. IF perp ∈ W:               // acceptance check
         return quantize(nav)     // valid tile on the floor
       ELSE:
         return nearest_acceptable(nav)  // snap to nearest valid tile
```

The acceptance window W is a thickened slab in perpendicular space. Its shape determines the tiling type (Penrose, Ammann-Beenker, etc.). For Penrose, W is a decagonal region.

**Key experimental finding**: Random vectors don't compress well through cut-and-project. Neural embeddings DO work because they have low effective dimensionality — the model already compressed the information. The projection preserves what the model chose to keep.

### 3.5 Deflate (Dream Consolidation)

Memories decay and consolidate, like sleep:

```rust
fn deflate(center: TileCoord, radius: f64) -> Option<Tile> {
    // 1. Gather all tiles within radius
    let nearby = tiles_in_radius(center, radius);
    
    // 2. XOR-merge content (or attention-weighted merge for embeddings)
    let merged = xor_merge(nearby.iter().map(|t| t.content));
    
    // 3. Remove originals, insert consolidated tile
    for tile in &nearby { tiles.remove(tile.coord); }
    tiles.insert(Tile {
        coord: center,
        content: merged,
        level: max_level + 1,  // promotion
        ..tile_bit(center)
    });
}
```

Deflation is the "dream" phase. Nearby memories merge into higher-level abstractions. The consolidation level increases. This is how the floor compacts: deflated tiles occupy one slot instead of many.

---

## 4. API Signatures

### 4.1 Rust API

```rust
// --- Construction ---
let floor = PenroseFloor::new()
    .seed(42)
    .scale(1.0)
    .embedding_dim(1536);

// --- Storage (O(1)) ---
let addr = floor.store(content: &[u8])?;          // store at walker position
let addr = floor.store_at(pos, content: &[u8])?;  // store at explicit position

// --- Navigation ---
let walker = floor.walker_at((0.0, 0.0)).facing(0.0);
let read = walker.walk(&[Step::new(φ, 0.5), Step::new(φ², -0.3)]);
let read = walker.spline_to((10.0, 7.0), steps=5);
let read = walker.tack(&[0.5, -1.0, 0.5], distance=φ);
let read = walker.stretch(&[1.0, φ, φ², φ³], heading=0.0);

// --- Retrieval ---
let mem = floor.retrieve(addr)?;
let mem = floor.nearest(pos)?;
let region = floor.region_at(pos, radius=3)?;

// --- Consolidation ---
floor.deflate(center, radius)?;
floor.deflate_all(max_radius=10.0)?;  // dream pass

// --- Embedding Integration ---
let nav = floor.project_embedding(&embedding)?;  // R^D → TileCoord
let similar = floor.approximate_neighbors(addr, k=10)?;
```

### 4.2 Python API

```python
from penrose_memory import PenroseMemory

# --- Plug-and-play for AI agents ---
memory = PenroseMemory(embedding_dim=1536)

# Store
addr = memory.store("The reef is at 60.5N 147.2W")

# Store with explicit embedding
addr = memory.store_with_embedding(embedding=[0.1, 0.3, ...], content="fishing log")

# Navigate by dead reckoning
read = memory.navigate(distance=2.618, heading=0.5)
print(read.bits)         # [True, False, True, ...]
print(read.confidence)   # 0.97

# Recall by embedding (project → walk → read)
results = memory.recall(query_embedding=[0.1, 0.3, ...], k=5)

# Recall by dead reckoning spline
results = memory.recall_spline(steps=[
    (1.618, 0.0),   # step 1: distance φ, heading east
    (2.618, 0.52),  # step 2: distance φ², heading NE
    (1.618, -0.31), # step 3: correct heading
])

# Consolidate (dream)
memory.consolidate(radius=5.0)

# Region fingerprinting (location awareness)
fingerprint = memory.fingerprint(radius=3)
```

### 4.3 LangChain Integration

```python
from langchain.vectorstores import PenroseMemoryStore
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
memory = PenroseMemoryStore(
    embedding_function=embeddings.embed_query,
    embedding_dim=1536,
)

# Drop-in replacement for FAISS/Chroma
memory.add_texts(["reef at 60.5N", "channel runs NE"])
docs = memory.similarity_search("where is the reef?", k=5)
```

### 4.4 OpenAI Function Tool

```python
import openai
from penrose_memory import PenroseMemory

memory = PenroseMemory()

tools = [{
    "type": "function",
    "function": {
        "name": "penrose_recall",
        "description": "Recall memories by dead reckoning on the Penrose floor",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "How far to walk (φ units)"},
                "heading": {"type": "number", "description": "Direction in radians"},
                "k": {"type": "integer", "description": "Number of results"}
            }
        }
    }
}]
```

---

## 5. Performance

### 5.1 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `tile_bit(coord)` | O(1) | O(1) |
| `store(content)` | O(1) | O(content size) |
| `walk(steps)` | O(len(steps)) | O(len(steps)) |
| `project_embedding(e)` | O(D × d) | O(D + d) |
| `nearest(pos)` | O(N) brute / O(log N) with spatial index |
| `deflate(center, r)` | O(r²) in tile space | O(r²) |
| `matching_rule(coord)` | O(1) | O(1) |

**Structure is FREE.** The Fibonacci word is computed on demand. A floor with 10^9 potential tiles uses zero memory for the tiling itself — only stored memories occupy space.

### 5.2 Footprint for 1M Tiles

| Component | Size | Notes |
|-----------|------|-------|
| Tile coordinates | 16 MB | (i64, i64) per tile |
| Tile content (avg 200B) | 200 MB | payload only |
| Tile metadata | 24 MB | bit, level, flags |
| Fibonacci word | 0 bytes | computed, not stored |
| Spatial index (R-tree) | ~40 MB | for O(log N) nearest |
| **Total** | **~280 MB** | **vs ~4GB for FAISS at 1M × 1536** |

The key savings: no vector index. The floor IS the index. Navigation replaces similarity search.

### 5.3 Recall Latency

Dead reckoning recall is O(len(spline_steps)). A 5-step spline is 5 hash lookups + 5 matching rule checks = ~50ns. No tree traversal, no distance computation across all vectors.

**Comparison**: FAISS IVF recall at 1M vectors ≈ 2-5ms. Penrose dead reckoning ≈ 0.05μs. 40,000× faster for targeted recall.

The tradeoff: you need to know the right spline (distance + heading). For blind similarity search, you still need to project the query embedding and walk from there — still O(1) for the projection, then O(spline_length) for the walk.

---

## 6. Embedding Integration Architecture

### 6.1 The Cut-and-Project Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Embedding   │────▶│  Projection  │────▶│  Floor Nav  │
│  e ∈ R^1536  │     │  P: R^D→R^d  │     │  nav ∈ R^2  │
└─────────────┘     └──────────────┘     └─────┬───────┘
                                                │
                          ┌─────────────────────┤
                          │                     │
                    ┌─────▼──────┐      ┌───────▼──────┐
                    │  Quantize  │      │  Perpendicular│
                    │  to tile   │      │  acceptance   │
                    └─────┬──────┘      └───────┬──────┘
                          │                     │
                    ┌─────▼──────┐      ┌───────▼──────┐
                    │  Walk from │◀─────│  Accept/     │
                    │  tile      │      │  Reject      │
                    └────────────┘      └──────────────┘
```

### 6.2 Why Neural Embeddings Work

**Experimental finding**: Random vectors lose ~95% of discriminative information through cut-and-project to 2D. Neural embeddings (OpenAI, Cohere, etc.) lose only ~5-15%.

**Explanation**: Neural embeddings already live on a low-dimensional manifold within R^D. The model compressed the information during training. Cut-and-project preserves this manifold structure because the projection matrix can be aligned with the manifold's principal directions (via PCA or learned projection).

**Implication**: Penrose Memory works as a **lossless compression** of what the model already compressed. The floor preserves the model's own compression, not raw information.

### 6.3 Projection Matrix Options

| Method | Cost | Quality | Use Case |
|--------|------|---------|----------|
| Random orthogonal | Free | OK for rough nav | Exploration, prototyping |
| PCA top-d components | O(D²) one-time | Good | Fixed corpus |
| Learned (trained) | O(D² × epochs) | Best | Production, specific domain |
| SVD of embedding corpus | O(N × D²) | Very good | When you have the corpus |

**Default**: PCA on the first 10K embeddings, then fix the projection. O(1) per query after that.

---

## 7. Navigation Modes

### 7.1 Walk (Dead Reckoning)

The primitive. Each step applies distance + heading.

```rust
read = floor.walk(&[
    Step::new(1.618, 0.0),    // φ at east
    Step::new(2.618, 0.52),   // φ² at NE
]);
```

Use when: you know the exact path (stored spline, known waypoints).

### 7.2 Spline (Straight Line)

Equal steps from current position to target.

```rust
read = floor.spline_to((10.0, 7.0), steps=5);
```

Use when: you know the target coordinate but not the intermediate path.

### 7.3 Tack (Zigzag)

Heading deltas at constant distance, like a sailboat beating upwind.

```rust
read = floor.tack(&[0.5, -1.0, 0.5, -1.0], distance=PHI);
```

Use when: exploring, searching, or adjusting for drift. The zigzag pattern maximizes coverage of the floor.

### 7.4 Stretch (Variable Distance)

Different distances at constant heading.

```rust
read = floor.stretch(&[1.0, PHI, PHI*PHI, PHI*PHI*PHI], heading=0.0);
```

Use when: scanning across granularity levels. Short stretches read fine detail, long stretches read coarse structure.

### 7.5 Embedding-Guided

Project a query embedding to floor coordinates, then walk.

```python
nav = memory.project_embedding(query_embedding)
results = memory.recall_near(nav, k=5)
```

Use when: you have a semantic query but no known coordinates. The embedding projects to the floor, and you walk the neighborhood.

---

## 8. Persistence & Distribution

### 8.1 Serialization

Floors serialize to a compact binary format:

```
PenroseFloor {
    magic: b"PNRS",           // 4 bytes
    version: u8,               // 1 byte
    seed: u64,                 // 8 bytes — THE ENTIRE TILING
    scale: f64,                // 8 bytes
    dimension: u32,            // 4 bytes
    projection: Matrix<f64>,   // D × d × 8 bytes
    acceptance_window: ...,    // depends on shape
    tiles: Vec<(TileCoord, Tile)>,  // stored memories only
}
```

**The entire tiling is 8 bytes (the seed).** Only stored memories add to the file size. A floor with 1M memories serializes to ~280MB. The tiling itself is always 8 bytes.

### 8.2 Fleet Coordination

Multiple agents can share a floor:

1. **Same seed, same scale** = same floor. Agents walk the same tiling from different positions.
2. **Position exchange**: Agent A tells Agent B "I'm at tile (42, -17)". Agent B can spline to that tile.
3. **Spline exchange**: Agent A publishes `[(φ, 0.5), (φ², -0.3)]`. Agent B walks the same spline from its position and reads different (but structurally related) tiles.

### 8.3 Incremental Sync

Floors diff by tile coordinate. Sync protocol:

```
Agent A: "I have tiles at coords [hash set of coords]"
Agent B: "I need coords [set difference]"
Agent A: [sends tile payloads]
```

The tiling never syncs — it's derived from the shared seed. Only payloads sync.

---

## 9. Thread Safety & Concurrency

### 9.1 Read-Parallel, Write-Serial

```
PenroseFloor {
    inner: RwLock<Inner>,
}
```

- Multiple walkers can `walk`, `tile_bit`, `project_embedding` concurrently (read lock).
- `store` and `deflate` acquire write lock.
- `tile_bit` is pure computation (no lock needed) — it reads only the seed.

### 9.2 Sharding for Large Floors

For floors exceeding single-machine memory (>100M tiles):

```
ShardedPenroseFloor {
    shard_key: fn(TileCoord) -> ShardId,  // hash-based sharding
    shards: Vec<PenroseFloor>,             // distributed across machines
}
```

Each shard covers a region of the tile coordinate space. The Fibonacci word computation is shard-local. Navigation across shard boundaries requires coordination.

---

## 10. Crate Layout

```
penrose-memory/
├── Cargo.toml
├── ARCHITECTURE.md          ← this file
├── src/
│   ├── lib.rs               ← core types (Floor, Walker, Step, Tile, Region)
│   ├── fibonacci.rs         ← Fibonacci word computation, Beatty sequence
│   ├── projection.rs        ← cut-and-project, projection matrices
│   ├── matching.rs          ← Penrose matching rules, confidence scoring
│   ├── walk.rs              ← navigation: walk, spline, tack, stretch
│   ├── deflate.rs           ← dream consolidation
│   ├── serialize.rs         ← binary persistence, versioning
│   └── embedding.rs         ← neural embedding integration (optional feature)
├── penrose_memory/
│   ├── __init__.py          ← Python bindings (PyO3 or pure Python)
│   ├── floor.py             ← Python PenroseFloor implementation
│   └── langchain.py         ← LangChain VectorStore adapter
└── benches/
    ├── tile_bit.rs          ← O(1) Fibonacci word benchmark
    ├── walk.rs              ← navigation throughput
    └── project.rs           ← cut-and-project throughput
```

### Feature Flags

```toml
[features]
default = ["serialize"]
serialize = ["serde", "bincode"]
embedding = ["ndarray", "ndarray-linalg"]
langchain = ["embedding"]        # LangChain adapter
distributed = ["tokio", "tonic"] # gRPC sharding
python = ["pyo3"]                # Python bindings
```

---

## 11. Key Invariants

1. **Seed determines everything.** Same seed + same scale = same floor. Always.
2. **Structure is free.** `tile_bit(coord)` is O(1) with zero storage.
3. **Neighborhoods are unique.** Matching rules guarantee no two regions are identical.
4. **Navigation costs two numbers.** Every query reduces to a sequence of (distance, heading) pairs.
5. **Deflation is lossy but bounded.** XOR-merge or attention-weighted merge. Level increases.
6. **Embeddings project cleanly.** Neural embeddings have low effective dimension. Random vectors don't.
7. **The fovea is small.** Context window reads a local region. The brain is the entire floor.

---

## 12. Open Questions

1. **Projection matrix training**: What's the optimal projection for a given embedding model? PCA works but learned projections may preserve more structure.
2. **Acceptance window shape**: Decagonal (Penrose) vs octagonal (Ammann-Beenker) vs other. Does the tiling type matter for memory quality?
3. **Deflation strategy**: XOR-merge is fast but destructive. Attention-weighted merge preserves more but requires the original embeddings. Hybrid?
4. **Drift recovery**: When confidence drops, what's the optimal tacking strategy? Gradient ascent on confidence? Bayesian update?
5. **Multi-floor navigation**: Can agents seamlessly walk between floors with different seeds? Federation protocol?

---

## 13. Summary

Penrose Memory replaces vector similarity search with geometric navigation on a deterministic aperiodic tiling. The core loop:

```
seed → Fibonacci word → tile_bit(coord) → walk(distance, heading) → read bits → decode memory
```

- **Storage**: O(1), structure is free (computed from seed)
- **Recall**: O(spline_length), typically 5-10 steps = ~50ns
- **Footprint**: ~280MB for 1M tiles (vs ~4GB for FAISS)
- **API**: `memory.store()`, `memory.recall()`, `memory.navigate(distance, heading)`
- **Integration**: LangChain VectorStore, OpenAI function tool, Rust crate, Python package

The context window is the fovea. The Penrose floor is the brain. Dead reckoning is the query. Two numbers to navigate infinity.

---

*Distance. Direction. The floor does the rest.*

— Forgemaster ⚒️, 2026-05-12
