"""Tests for penrose-memory."""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from penrose_memory import PenroseMemory


# ─── Construction ─────────────────────────────────────────────────────────────

def test_construction():
    pm = PenroseMemory(embedding_dim=128)
    assert len(pm) == 0


def test_default_embedding_dim():
    pm = PenroseMemory()
    assert pm._embedding_dim == 1536


# ─── Store ─────────────────────────────────────────────────────────────────────

def test_store_returns_id():
    pm = PenroseMemory(embedding_dim=4)
    tid = pm.store("hello", [0.1, 0.2, 0.3, 0.4])
    assert tid == 1
    assert len(pm) == 1


def test_store_multiple():
    pm = PenroseMemory(embedding_dim=4)
    id1 = pm.store("first", [1.0, 0.0, 0.0, 0.0])
    id2 = pm.store("second", [0.0, 1.0, 0.0, 0.0])
    id3 = pm.store("third", [0.0, 0.0, 1.0, 0.0])
    assert id1 == 1
    assert id2 == 2
    assert id3 == 3
    assert len(pm) == 3


# ─── Recall ────────────────────────────────────────────────────────────────────

def test_recall_exact_match():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("hello", [0.1, 0.2, 0.3, 0.4])
    results = pm.recall([0.1, 0.2, 0.3, 0.4])
    assert len(results) == 1
    assert results[0]["text"] == "hello"
    assert results[0]["confidence"] > 0.9


def test_recall_empty():
    pm = PenroseMemory(embedding_dim=4)
    results = pm.recall([0.1, 0.2, 0.3, 0.4])
    assert results == []


def test_recall_sorted_by_confidence():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("far", [10.0, 20.0, 30.0, 40.0])
    pm.store("near", [0.1, 0.2, 0.3, 0.4])
    results = pm.recall([0.1, 0.2, 0.3, 0.4])
    assert results[0]["text"] == "near"
    assert results[0]["confidence"] > results[1]["confidence"]


def test_recall_different_embeddings_prefer_closest():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("A", [1.0, 0.0, 0.0, 0.0])
    pm.store("B", [0.0, 0.0, 0.0, 1.0])
    r1 = pm.recall([1.0, 0.0, 0.0, 0.0])
    r2 = pm.recall([0.0, 0.0, 0.0, 1.0])
    assert r1[0]["text"] == "A"
    assert r2[0]["text"] == "B"


# ─── Projection ────────────────────────────────────────────────────────────────

def test_projection_2d():
    pm = PenroseMemory(embedding_dim=4)
    x, y = pm._project_to_2d([1.0, 2.0, 3.0, 4.0])
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_projection_deterministic():
    pm = PenroseMemory(embedding_dim=4)
    emb = [0.5, 1.0, 1.5, 2.0]
    p1 = pm._project_to_2d(emb)
    p2 = pm._project_to_2d(emb)
    assert p1 == p2


def test_projection_different_embeddings():
    pm = PenroseMemory(embedding_dim=4)
    p1 = pm._project_to_2d([1.0, 0.0, 0.0, 0.0])
    p2 = pm._project_to_2d([0.0, 0.0, 0.0, 1.0])
    assert p1 != p2


# ─── Fibonacci Word ────────────────────────────────────────────────────────────

def test_fibonacci_ratio():
    pm = PenroseMemory(embedding_dim=4)
    bits = [pm._tile_bit(q, 0) for q in range(10000)]
    ratio = sum(bits) / len(bits)
    inv_phi = 0.618033988749895
    assert abs(ratio - inv_phi) < 0.03


def test_three_coloring():
    pm = PenroseMemory(embedding_dim=4)
    colors = set(pm._three_color(q, 0) for q in range(300))
    assert colors == {0, 1, 2}


# ─── Navigate ──────────────────────────────────────────────────────────────────

def test_navigate_zero_distance():
    pm = PenroseMemory(embedding_dim=4)
    tid = pm.store("nav", [1.0, 0.0, 0.0, 0.0])
    result = pm.navigate(tid, 0.0, 0.0)
    assert tid in result


def test_navigate_nonexistent():
    pm = PenroseMemory(embedding_dim=4)
    result = pm.navigate(999, 1.0, 0.0)
    assert result == []


# ─── Consolidate ───────────────────────────────────────────────────────────────

def test_consolidate_nearby():
    pm = PenroseMemory(embedding_dim=4)
    for i in range(20):
        val = 1.0 + i * 0.001
        pm.store(f"mem-{i}", [val, 2.0, 3.0, 4.0])
    before = len(pm)
    removed = pm.consolidate()
    assert len(pm) < before


def test_consolidate_empty():
    pm = PenroseMemory(embedding_dim=4)
    assert pm.consolidate() == 0


def test_consolidate_single():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("solo", [1.0, 2.0, 3.0, 4.0])
    assert pm.consolidate() == 0


# ─── Confidence & Distance ─────────────────────────────────────────────────────

def test_confidence_decreases_with_distance():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("center", [1.0, 2.0, 3.0, 4.0])
    close = pm.recall([1.0, 2.0, 3.0, 4.0])
    far = pm.recall([10.0, 20.0, 30.0, 40.0])
    assert close[0]["confidence"] > far[0]["confidence"]


def test_result_has_heading():
    pm = PenroseMemory(embedding_dim=4)
    pm.store("test", [1.0, 2.0, 3.0, 4.0])
    results = pm.recall([5.0, 6.0, 7.0, 8.0])
    assert "heading" in results[0]
    assert "distance" in results[0]


# ─── Large Embedding ───────────────────────────────────────────────────────────

def test_large_embedding():
    pm = PenroseMemory(embedding_dim=1536)
    emb = [math.sin(i) * 0.1 for i in range(1536)]
    tid = pm.store("big", emb)
    results = pm.recall(emb)
    assert results[0]["text"] == "big"
