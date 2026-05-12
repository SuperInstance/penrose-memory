# Penrose Memory

Aperiodic memory palace for AI agents. Navigate memories by **distance + direction** on a Penrose floor.

## Install

```toml
[dependencies]
penrose-memory = "0.1.0"
```

## Quick Start

```rust
use penrose_memory::{PenroseFloor, Step};

// Create a memory floor
let mut floor = PenroseFloor::new()
    .at(0.0, 0.0)    // start position
    .facing(0.0);     // facing east

// Store memories
floor.store_here(0xDEADBEEF);
floor.store_at(10.0, 5.0, 0xCAFEBABE);

// Navigate by dead reckoning
let steps = vec![Step::new(10.0, 0.46)]; // distance + heading
let read = floor.walk(&steps);
println!("Confidence: {}", read.confidence);

// Spline directly to a target
let read = floor.spline_to((10.0, 5.0), 5);

// Deflate (consolidate nearby memories)
floor.deflate((0.0, 0.0), 3.0);
```

## Why Penrose?

| Vector DB | Penrose Floor |
|---|---|
| All neighborhoods identical | Every neighborhood unique |
| Scalar distance retrieval | Bragg peak retrieval (structured) |
| Hash collisions possible | Zero collisions (matching rules) |
| Artificial hierarchy | Golden hierarchy (φ^k) |
| Fixed context window | Self-similar (grows with zoom) |

## Navigation Primitives

| Method | Description |
|---|---|
| `walk(steps)` | Dead reckoning: read one bit per step |
| `spline_to(target, n)` | Straight-line walk to target |
| `tack(deltas, dist)` | Zigzag like a sailboat |
| `stretch(stretches, heading)` | Varying distances at constant heading |
| `deflate(center, radius)` | Consolidate nearby memories (dream) |

## The Math

The floor uses the **Fibonacci word** — the same sequence that determines thick/thin tiles in a Penrose tiling. The ratio of thick to thin converges to 1/φ ≈ 0.618. The pattern is deterministic: once you see the local pattern lock in, there's only one way it can continue.

This is **dead reckoning**: the ancient navigator's technique. No GPS. No absolute coordinates. Just "how far" and "which way." The floor pattern confirms you're on the right path.

## License

MIT
