"""Advanced tests for penrose-memory — encoding edge cases, retrieval, decay, navigate, consolidate."""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from penrose_memory import PenroseMemory, PHI, INV_PHI, GOLDEN_ANGLE


# ─── Constants ─────────────────────────────────────────────────────────────────

def test_phi_constant():
    """Golden ratio is correct."""
    assert abs(PHI - 1.618033988749895) < 1e-12


def test_inv_phi_constant():
    """Inverse golden ratio is correct."""
    assert abs(INV_PHI - 0.618033988749895) < 1e-12
    assert abs(INV_PHI - 1/PHI) < 1e-12


def test_golden_angle_constant():
    """Golden angle is correct."""
    expected = math.pi * (3 - math.sqrt(5))
    assert abs(GOLDEN_ANGLE - expected) < 1e-12


# ─── Projection Edge Cases ────────────────────────────────────────────────────

def test_projection_zero_embedding():
    """All-zero embedding projects to origin."""
    pm = PenroseMemory(embedding_dim=4)
    x, y = pm._project_to_2d([0.0, 0.0, 0.0, 0.0])
    assert x == 0.0
    assert y == 0.0


def test_projection_single_dim():
    """1D embedding projects correctly."""
    pm = PenroseMemory(embedding_dim=1)
    x, y = pm._project_to_2d([5.0])
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_projection_shorter_than_dim():
    """Embedding shorter than declared dim gets padded with 0."""
    pm = PenroseMemory(embedding_dim=8)
    x, y = pm._project_to_2d([1.0, 2.0])  # only 2 values, dim=8
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_projection_large_values():
    """Large embedding values don't cause overflow."""
    pm = PenroseMemory(embedding_dim=4)
    x, y = pm._project_to_2d([1e6, 1e6, 1e6, 1e6])
    assert math.isfinite(x)
    assert math.isfinite(y)


def test_projection_negative_values():
    """Negative values use abs, so projection is symmetric in magnitude."""
    pm = PenroseMemory(embedding_dim=4)
    x1, y1 = pm._project_to_2d([1.0, 2.0, 3.0, 4.0])
    x2, y2 = pm._project_to_2d([-1.0, -2.0, -3.0, -4.0])
    # abs() makes them identical
    assert abs(x1 - x2) < 1e-10
    assert abs(y1 - y2) < 1e-10


# ─── Quantization ─────────────────────────────────────────────────────────────

def test_quantization_rounds_to_phi_grid():
    """Quantize maps to lattice positions."""
    pm = PenroseMemory(embedding_dim=4)
    qx, qy = pm._quantize(PHI, PHI)
    assert qx == 1
    assert qy == 1


def test_quantization_zero():
    """Origin quantizes to (0, 0)."""
    pm = PenroseMemory(embedding_dim=4)
    qx, qy = pm._quantize(0.0, 0.0)
    assert qx == 0
    assert qy == 0


# ─── Fibonacci Word / Tile Bit ────────────────────────────────────────────────

def test_tile_bit_deterministic():
    """Same position always returns same bit."""
    pm = PenroseMemory(embedding_dim=4)
    for _ in range(10):
        assert pm._tile_bit(5, 10) == pm._tile_bit(5, 10)


def test_tile_bit_symmetry_broken():
    """Swapping coords generally gives different bits (not guaranteed but likely)."""
    pm = PenroseMemory(embedding_dim=4)
    different = sum(1 for q in range(100) if pm._tile_bit(q, q+1) != pm._tile_bit(q+1, q))
    assert different > 10  # Most should differ


# ─── 3-Coloring ───────────────────────────────────────────────────────────────

def test_three_coloring_values():
    """3-coloring only returns 0, 1, or 2."""
    pm = PenroseMemory(embedding_dim=4)
    for x in range(-50, 50):
        for y in range(-5, 5):
            c = pm._three_color(x, y)
            assert c in (0, 1, 2)


def test_three_coloring_deterministic():
    """Same position always same color."""
    pm = PenroseMemory(embedding_dim=4)
    assert pm._three_color(7, 13) == pm._three_color(7, 13)


# ─── Store Edge Cases ─────────────────────────────────────────────────────────

def test_store_same_embedding_twice():
    """Storing same embedding twice creates two tiles."""
    pm = PenroseMemory(embedding_dim=4)
    emb = [1.0, 2.0, 3.0, 4.0]
    id1 = pm.store("first", emb)
    id2 = pm.store("second", emb)
    assert id1 != id2
    assert len(pm) == 2


def test_store_empty_text():
    """Empty string is valid."""
    pm = PenroseMemory(embedding_dim=4)
    tid = pm.store("", [1.0, 2.0, 3.0, 4.0])
    assert tid == 1
    results = pm.recall([1.0, 2.0, 3.0, 4.0])
    assert results[0]["text"] == ""


def test_store_long_text():
    """Very long text stored correctly."""
    pm = PenroseMemory(embedding_dim=4)
    long_text = "x" * 10000
    tid = pm.store(long_text, [1.0, 2.0, 3.0, 4.0])
    results = pm.recall([1.0, 2.0, 3.0, 4.0])
    assert results[0]["text"] == long_text


def test_store_sequential_ids():
    """IDs increment properly."""
    pm = PenroseMemory(embedding_dim=4)
    ids = [pm.store(f"m{i}", [float(i)]*4) for i in range(10)]
    assert ids == list(range(1, 11))


# ─── Recall Edge Cases ────────────────────────────────────────────────────────

def test_recall_max_steps_zero():
    """max_steps=0 skips path verification."""
    pm = PenroseMemory(embedding_dim=4)
    pm.store("test", [1.0, 2.0, 3.0, 4.0])
    results = pm.recall([1.0, 2.0, 3.0, 4.0], max_steps=0)
    assert len(results) == 1
    assert results[0]["confidence"] > 0.9


def test_recall_many_stored():
    """Recall from many stored memories."""
    pm = PenroseMemory(embedding_dim=4)
    for i in range(100):
        pm.store(f"mem-{i}", [float(i), float(i+1), float(i+2), float(i+3)])
    results = pm.recall([50.0, 51.0, 52.0, 53.0])
    assert len(results) == 100
    assert results[0]["text"] == "mem-50"


def test_recall_result_fields():
    """All expected fields present."""
    pm = PenroseMemory(embedding_dim=4)
    pm.store("test", [1.0, 2.0, 3.0, 4.0])
    r = pm.recall([5.0, 6.0, 7.0, 8.0])[0]
    assert "tile_id" in r
    assert "text" in r
    assert "confidence" in r
    assert "distance" in r
    assert "heading" in r


# ─── Navigate Advanced ────────────────────────────────────────────────────────

def test_navigate_to_another_tile():
    """Navigate from one tile toward another."""
    pm = PenroseMemory(embedding_dim=4)
    tid1 = pm.store("origin", [1.0, 0.0, 0.0, 0.0])
    tid2 = pm.store("target", [0.0, 1.0, 0.0, 0.0])
    # Get distance between tiles
    x1, y1 = pm._project_to_2d([1.0, 0.0, 0.0, 0.0])
    x2, y2 = pm._project_to_2d([0.0, 1.0, 0.0, 0.0])
    dist = math.hypot(x2-x1, y2-y1)
    heading = math.atan2(y2-y1, x2-x1)
    found = pm.navigate(tid1, dist, heading)
    assert tid2 in found


def test_navigate_large_distance():
    """Navigate very far — likely no tiles found."""
    pm = PenroseMemory(embedding_dim=4)
    tid = pm.store("only", [1.0, 0.0, 0.0, 0.0])
    found = pm.navigate(tid, 1000.0, 0.0)
    assert tid not in found  # Too far from itself


# ─── Consolidate Advanced ─────────────────────────────────────────────────────

def test_consolidate_distant_tiles_not_merged():
    """Tiles far apart are NOT consolidated."""
    pm = PenroseMemory(embedding_dim=4)
    pm.store("a", [1.0, 0.0, 0.0, 0.0])
    pm.store("b", [100.0, 0.0, 0.0, 0.0])
    before = len(pm)
    removed = pm.consolidate()
    assert removed == 0
    assert len(pm) == before


def test_consolidate_preserves_level_above_zero():
    """Tiles with level > 0 are skipped during consolidation."""
    pm = PenroseMemory(embedding_dim=4)
    pm.store("a", [1.0, 2.0, 3.0, 4.0])
    # Manually set level
    pm._tiles[0]["level"] = 1
    assert pm.consolidate() == 0


def test_consolidate_returns_removed_count():
    """Consolidate returns number of tiles removed."""
    pm = PenroseMemory(embedding_dim=4)
    for i in range(20):
        pm.store(f"m{i}", [1.0 + i * 0.001, 2.0, 3.0, 4.0])
    removed = pm.consolidate()
    assert removed > 0
    assert len(pm) < 20  # some were merged


# ─── Euclidean Distance ───────────────────────────────────────────────────────

def test_euclidean_distance_same_point():
    """Distance from a point to itself is 0."""
    pm = PenroseMemory(embedding_dim=4)
    assert pm._euclidean_distance(1.0, 2.0, 1.0, 2.0) == 0.0


def test_euclidean_distance_unit():
    """Known distance."""
    pm = PenroseMemory(embedding_dim=4)
    assert abs(pm._euclidean_distance(0, 0, 3, 4) - 5.0) < 1e-10


# ─── Heading ──────────────────────────────────────────────────────────────────

def test_heading_east():
    """Heading due east is 0."""
    pm = PenroseMemory(embedding_dim=4)
    assert abs(pm._heading_to(0, 0, 1, 0)) < 1e-10


def test_heading_north():
    """Heading due north is π/2."""
    pm = PenroseMemory(embedding_dim=4)
    assert abs(pm._heading_to(0, 0, 0, 1) - math.pi/2) < 1e-10
