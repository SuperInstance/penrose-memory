"""
Tests for penrose_memory Python package.
Run: python -m pytest penrose_memory/tests.py -v
"""

import math
from . import PenroseFloor, Step, PHI


def test_store_and_retrieve():
    floor = PenroseFloor()
    floor.store(5.0, 5.0, "test_memory")
    assert floor.retrieve(5.0, 5.0) == "test_memory"


def test_dead_reckoning_walk():
    floor = PenroseFloor(x=0.0, y=0.0, heading=0.0)
    floor.store_here("origin")
    steps = [Step(5.0, 0.0)]
    read = floor.walk(steps)
    assert len(read.bits) == 1
    assert read.confidence > 0.0


def test_spline():
    floor = PenroseFloor()
    floor.store(0.0, 0.0, "start")
    read = floor.spline_to((10.0, 0.0), steps=5)
    assert len(read.bits) == 5
    assert read.confidence > 0.0


def test_tack():
    floor = PenroseFloor(heading=0.0)
    read = floor.tack([0.5, -1.0, 0.5, -1.0], distance=PHI)
    assert len(read.headings) == 4


def test_stretch():
    floor = PenroseFloor(heading=0.0)
    read = floor.stretch([1.0, PHI, 1.0], heading=0.0)
    assert len(read.bits) == 3


def test_deflate():
    floor = PenroseFloor()
    floor.store(0.0, 0.0, "a")
    floor.store(1.0, 0.0, "b")
    floor.store(0.0, 1.0, "c")
    result = floor.deflate((0.0, 0.0), 3.0)
    assert result is not None
    assert len(result) == 3
    assert len(floor) == 1  # Consolidated


def test_nearest():
    floor = PenroseFloor()
    floor.store(0.0, 0.0, "near")
    floor.store(100.0, 100.0, "far")
    assert floor.nearest(1.0, 1.0) == "near"


def test_aperiodic_bits():
    floor = PenroseFloor()
    east = [floor._tile_bit((q, 0)) for q in range(50)]
    north = [floor._tile_bit((0, r)) for r in range(50)]
    assert east != north, "Different directions should give different patterns"


def test_fibonacci_ratio():
    floor = PenroseFloor()
    bits = [floor._tile_bit((q, 0)) for q in range(1000)]
    ratio = sum(bits) / len(bits)
    assert abs(ratio - 1.0/PHI) < 0.05, f"Ratio {ratio} should approach 1/φ"


def test_empty_retrieve():
    floor = PenroseFloor()
    assert floor.retrieve(999.0, 999.0) is None
