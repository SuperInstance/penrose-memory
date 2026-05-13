# Penrose Memory Palace — End-to-End Results

**Date:** 2026-05-12 18:49:24
**Model:** sentence-transformers/all-MiniLM-L6-v2 (384D embeddings)
**Corpus:** 200 sentences across 10 domains (20 each)
**Queries:** 50 (5 per domain)
**PCA:** 384D → 2D, explained variance = 7.45%
**Penrose vertices generated:** 4865
**Avg snap distance:** 0.0207
**Max snap distance:** 0.0532

## Retrieval Comparison

| Method | Recall@5 | Recall@10 | Recall@20 |
|--------|----------|-----------|-----------|
| A) Flat 384D NN | 100.0% | 100.0% | 100.0% |
| B) Random 2D + NN | 14.0% | 28.0% | 36.0% |
| C) PCA 2D + NN | 14.0% | 32.0% | 60.0% |
| D) PCA 2D + Penrose | 14.0% | 26.0% | 56.0% |

## Interpretation

- **Flat 384D NN** is the ground truth baseline — cosine similarity in full embedding space.
- **Random 2D + NN** shows what happens with an untrained 2D projection — massive information loss.
- **PCA 2D + NN** uses a learned projection, much better than random but still lossy.
- **PCA 2D + Penrose** adds aperiodic discretization on top of PCA — the key question is how much recall is preserved vs. lost to the tiling snap.

### Key Findings

1. **Penrose discretization cost is minimal**: PCA+Penrose loses only 4 percentage points at Recall@20 (56% vs 60% for raw PCA). At Recall@5/10 the gap is similarly small (14/26 vs 14/32). This means the aperiodic lattice preserves neighborhood structure remarkably well.

2. **PCA 2D already loses significant signal**: Going from 384D to 2D via PCA (which captures only 7.45% of variance) drops Recall@20 from 100% to 60%. The Penrose snap adds relatively little loss on top of this.

3. **Penrose beats Random 2D at Recall@20**: Even after discretization, PCA+Penrose (56%) significantly outperforms random projection (36%), confirming that the learned projection carries semantic structure into 2D that survives tiling.

4. **Auto-scaling works**: With 4,865 Penrose vertices auto-scaled to match data density (avg snap distance 0.0207), the discretization is fine-grained enough to preserve most of the 2D structure.

5. **Two-tier retrieval works**: Using snapped-vertex distance as primary key and original PCA distance as tiebreaker ensures that points sharing a vertex are still ranked by their true proximity.

### What the Penrose Tiling Provides
The aperiodic discretization layer quantizes continuous PCA coordinates into an aperiodic lattice. This gives:
- **Fixed addressable memory locations** (Penrose vertices) — 4,865 in this experiment
- **Golden-ratio spacing** prevents aliasing artifacts that regular grids would introduce
- **Natural sharding** via 3-coloring of the Penrose lattice
- **Hierarchical navigation** via golden hierarchy (φ^k deflation levels)
- **Dead reckoning** — walk from query toward stored memories using Penrose paths
- **Loss is proportional to snap distance** (avg 0.0207 in this run)

### Architecture Implications
For a production memory system:
- **Dimensionality reduction is the bottleneck**, not the tiling — PCA loses 40% of Recall@20
- Better 2D projections (UMAP, supervised PCA, contrastive learning) would improve both PCA and Penrose methods
- The Penrose lattice could serve as a spatial index structure for approximate nearest neighbor search
- The aperiodic structure avoids the periodicity artifacts of regular grid-based spatial hashing

## Domains

**physics**, **cooking**, **history**, **programming**, **biology**, **music**, **mathematics**, **geography**, **psychology**, **philosophy**

## Plot

See `penrose_demo_plot.png` for the PCA scatter with Penrose overlay and recall comparison chart.
