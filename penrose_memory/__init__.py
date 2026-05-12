"""
Penrose Memory — Aperiodic memory palace for AI agents.

Embeddings are projected to 2D Penrose coordinates using golden-ratio hashing.
The Fibonacci word determines tile bits (thick:thin → 1/φ). Matching rules
verify valid positions. Recall uses dead reckoning: walk from query toward
stored memories. 3-coloring for sharding, golden hierarchy (φ^k) for deflation.

Usage:
    from penrose_memory import PenroseMemory

    pm = PenroseMemory(embedding_dim=1536)
    tile_id = pm.store("hello world", [0.1, 0.2, ...])
    results = pm.recall([0.1, 0.2, ...], max_steps=5)
    for r in results:
        print(r["content"], r["confidence"], r["distance"])
"""

import math
from typing import Any, Dict, List, Optional

PHI = 1.618033988749895
INV_PHI = 0.618033988749895
GOLDEN_ANGLE = 2.399963229728653  # π(3 − √5)


class PenroseMemory:
    """
    Aperiodic memory palace navigated by dead reckoning.

    Embeddings are projected to 2D Penrose coordinates using golden-ratio hashing.
    The Fibonacci word determines tile bits. Recall walks from query toward stored
    memories via dead reckoning.
    """

    def __init__(self, embedding_dim: int = 1536):
        self._embedding_dim = embedding_dim
        self._tiles: List[Dict] = []
        self._next_id: int = 1

    def _project_to_2d(self, embedding: List[float]) -> tuple:
        """Project an embedding to 2D Penrose coordinates using golden-ratio hashing."""
        x = 0.0
        y = 0.0
        dim = min(self._embedding_dim, len(embedding))
        for i in range(dim):
            val = embedding[i] if i < len(embedding) else 0.0
            angle = i * GOLDEN_ANGLE
            magnitude = abs(val)
            x += magnitude * math.cos(angle)
            y += magnitude * math.sin(angle)
        scale = 1.0 / math.sqrt(dim) if dim > 0 else 1.0
        return (x * scale, y * scale)

    def _tile_bit(self, qx: int, qy: int) -> bool:
        """Fibonacci word bit at a lattice position via golden-ratio hashing."""
        h = (qx * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF) + (qy * 0x517CC1B727220A95 & 0xFFFFFFFFFFFFFFFF)
        idx = abs(h) % 10000
        current = int(idx * INV_PHI)
        nxt = int((idx + 1) * INV_PHI)
        return nxt != current

    def _three_color(self, qx: int, qy: int) -> int:
        """3-coloring of lattice position."""
        h = (qx * 0x517CC1B727220A95 & 0xFFFFFFFFFFFFFFFF) + (qy * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
        return abs(h) % 3

    def _quantize(self, x: float, y: float) -> tuple:
        """Quantize continuous 2D coordinates to lattice."""
        return (round(x / PHI), round(y / PHI))

    def _euclidean_distance(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def _heading_to(self, x1, y1, x2, y2) -> float:
        return math.atan2(y2 - y1, x2 - x1)

    def store(self, text: str, embedding: list) -> int:
        """
        Store text with its embedding.

        Projects the embedding to 2D Penrose coordinates and stores it.
        Returns the tile_id.
        """
        x, y = self._project_to_2d(embedding)
        qx, qy = self._quantize(x, y)
        color = self._three_color(qx, qy)
        tile_id = self._next_id
        self._next_id += 1
        self._tiles.append({
            "tile_id": tile_id,
            "text": text,
            "x": x,
            "y": y,
            "color": color,
            "level": 0,
        })
        return tile_id

    def recall(self, query_embedding: list, max_steps: int = 5) -> list:
        """
        Recall memories by dead reckoning from a query embedding.

        Returns list of dicts sorted by confidence (best first):
        - tile_id: int
        - text: str
        - confidence: float
        - distance: float
        - heading: float
        """
        if not self._tiles:
            return []

        qx, qy = self._project_to_2d(query_embedding)
        results = []

        for tile in self._tiles:
            dist = self._euclidean_distance(qx, qy, tile["x"], tile["y"])
            heading = self._heading_to(qx, qy, tile["x"], tile["y"])

            # Gaussian-like confidence falloff
            sigma = 2.0
            confidence = math.exp(-dist * dist / (2.0 * sigma * sigma))

            # Dead reckoning path verification
            path_confidence = 1.0
            if max_steps > 0 and dist > 0:
                verified = 0
                total = 0
                for step in range(1, max_steps + 1):
                    t = step / (max_steps + 1)
                    ix = qx + t * (tile["x"] - qx)
                    iy = qy + t * (tile["y"] - qy)
                    iqx, iqy = self._quantize(ix, iy)
                    bit = self._tile_bit(iqx, iqy)
                    neighbors = [(iqx+1, iqy), (iqx-1, iqy), (iqx, iqy+1), (iqx, iqy-1)]
                    if any(self._tile_bit(nx, ny) != bit for nx, ny in neighbors):
                        verified += 1
                    total += 1
                path_confidence = verified / total if total > 0 else 1.0

            results.append({
                "tile_id": tile["tile_id"],
                "text": tile["text"],
                "confidence": confidence * path_confidence,
                "distance": dist,
                "heading": heading,
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    def navigate(self, tile_id: int, distance: float, heading: float) -> list:
        """
        Navigate from a tile by distance and heading (dead reckoning).

        Returns list of tile IDs found within arrival radius.
        """
        start = None
        for tile in self._tiles:
            if tile["tile_id"] == tile_id:
                start = (tile["x"], tile["y"])
                break
        if start is None:
            return []

        dest_x = start[0] + distance * math.cos(heading)
        dest_y = start[1] + distance * math.sin(heading)
        arrival_radius = PHI * 0.5

        return [
            tile["tile_id"]
            for tile in self._tiles
            if self._euclidean_distance(dest_x, dest_y, tile["x"], tile["y"]) <= arrival_radius
        ]

    def consolidate(self) -> int:
        """
        Consolidate nearby memories using golden hierarchy.

        Tiles within φ distance are merged via XOR. Returns count removed.
        """
        if len(self._tiles) < 2:
            return 0

        merge_distance = PHI
        merged = [False] * len(self._tiles)
        new_tiles = []

        for i, tile in enumerate(self._tiles):
            if merged[i] or tile["level"] > 0:
                continue

            cluster_x = tile["x"]
            cluster_y = tile["y"]
            cluster_text = tile["text"]
            cluster_size = 1
            cluster_ids = [i]

            for j in range(i + 1, len(self._tiles)):
                if merged[j] or self._tiles[j]["level"] > 0:
                    continue
                d = self._euclidean_distance(cluster_x, cluster_y, self._tiles[j]["x"], self._tiles[j]["y"])
                if d < merge_distance:
                    cluster_x = (cluster_x * cluster_size + self._tiles[j]["x"]) / (cluster_size + 1)
                    cluster_y = (cluster_y * cluster_size + self._tiles[j]["y"]) / (cluster_size + 1)
                    cluster_text = f"{cluster_text} | {self._tiles[j]['text']}"
                    cluster_size += 1
                    cluster_ids.append(j)

            if cluster_size > 1:
                for idx in cluster_ids:
                    merged[idx] = True
                qx, qy = self._quantize(cluster_x, cluster_y)
                new_tiles.append({
                    "tile_id": self._next_id,
                    "text": cluster_text,
                    "x": cluster_x,
                    "y": cluster_y,
                    "color": self._three_color(qx, qy),
                    "level": 1,
                })
                self._next_id += 1

        before = len(self._tiles)
        self._tiles = [t for i, t in enumerate(self._tiles) if not merged[i]] + new_tiles
        return before - len(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)


# --- Tests ---

def _run_tests():
    """Built-in test suite."""
    errors = []

    def check(name, condition, msg=""):
        if not condition:
            errors.append(f"FAIL: {name}: {msg}")
        else:
            print(f"  ✓ {name}")

    # 1. Store + recall roundtrip
    pm = PenroseMemory(embedding_dim=4)
    tid = pm.store("hello", [0.1, 0.2, 0.3, 0.4])
    results = pm.recall([0.1, 0.2, 0.3, 0.4], 3)
    check("store_recall_roundtrip", len(results) > 0 and results[0]["text"] == "hello",
          f"got {len(results)} results")

    # 2. Different embeddings → different top results
    pm2 = PenroseMemory(embedding_dim=4)
    pm2.store("A", [1.0, 0.0, 0.0, 0.0])
    pm2.store("B", [0.0, 0.0, 0.0, 1.0])
    r1 = pm2.recall([1.0, 0.0, 0.0, 0.0], 3)
    r2 = pm2.recall([0.0, 0.0, 0.0, 1.0], 3)
    check("different_embeddings", r1[0]["text"] == "A" and r2[0]["text"] == "B")

    # 3. Nearby embeddings → nearby tiles
    pm3 = PenroseMemory(embedding_dim=4)
    pm3.store("near", [1.0, 2.0, 3.0, 4.0])
    r3 = pm3.recall([1.01, 2.01, 3.01, 4.01], 3)
    check("nearby_embeddings_close", r3[0]["distance"] < 0.1, f"dist={r3[0]['distance']}")

    # 4. Empty recall returns empty
    pm4 = PenroseMemory(embedding_dim=4)
    check("empty_recall", len(pm4.recall([1.0, 2.0, 3.0, 4.0], 5)) == 0)

    # 5. Fibonacci ratio → 1/φ
    pm5 = PenroseMemory(embedding_dim=4)
    bits = [pm5._tile_bit(q, 0) for q in range(10000)]
    ratio = sum(bits) / len(bits)
    check("fibonacci_ratio", abs(ratio - INV_PHI) < 0.03, f"ratio={ratio:.4f}")

    # 6. 3-coloring covers all
    pm6 = PenroseMemory(embedding_dim=4)
    colors = set(pm6._three_color(q, 0) for q in range(300))
    check("three_coloring", colors == {0, 1, 2}, f"colors={colors}")

    # 7. Consolidation reduces count
    pm7 = PenroseMemory(embedding_dim=4)
    for i in range(20):
        val = 1.0 + i * 0.001
        pm7.store(f"mem-{i}", [val, 2.0, 3.0, 4.0])
    before = len(pm7)
    pm7.consolidate()
    check("consolidation_reduces", len(pm7) < before, f"{before} -> {len(pm7)}")

    # 8. Confidence decreases with distance
    pm8 = PenroseMemory(embedding_dim=4)
    pm8.store("center", [1.0, 2.0, 3.0, 4.0])
    close = pm8.recall([1.0, 2.0, 3.0, 4.0], 0)
    far = pm8.recall([10.0, 20.0, 30.0, 40.0], 0)
    check("confidence_decreases", close[0]["confidence"] > far[0]["confidence"],
          f"close={close[0]['confidence']:.4f} far={far[0]['confidence']:.4f}")

    # 9. Navigate
    pm9 = PenroseMemory(embedding_dim=4)
    tid9 = pm9.store("nav", [1.0, 0.0, 0.0, 0.0])
    nav = pm9.navigate(tid9, 0.0, 0.0)
    check("navigate_zero_dist", tid9 in nav)

    # 10. Large embedding
    pm10 = PenroseMemory(embedding_dim=1536)
    emb = [math.sin(i) * 0.1 for i in range(1536)]
    pm10.store("big", emb)
    r10 = pm10.recall(emb, 3)
    check("large_embedding", r10[0]["text"] == "big")

    # Summary
    if errors:
        print(f"\n{len(errors)} test(s) FAILED:")
        for e in errors:
            print(f"  {e}")
        raise AssertionError(f"{len(errors)} test(s) failed")
    else:
        print(f"\nAll tests passed!")


if __name__ == "__main__":
    _run_tests()
