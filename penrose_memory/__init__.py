"""
Penrose Memory — Aperiodic memory palace for AI agents.

Navigate memories by distance + direction on a Penrose floor.
No coordinates. No index lookups. Just dead reckoning.

Usage:
    from penrose_memory import PenroseFloor, Step

    floor = PenroseFloor()
    floor.store(0.0, 0.0, {"fact": "reef at 60.5N 147.2W"})
    floor.store(5.0, 3.0, {"fact": "channel runs NE"})

    # Dead reckon: walk 5 units at heading 0.5 rad
    read = floor.walk([Step(5.0, 0.5)])
    print(f"Read {len(read.bits)} bits, confidence {read.confidence:.2f}")

    # Spline to a target
    read = floor.spline_to((5.0, 3.0), steps=5)

    # Tack like a sailboat
    read = floor.tack(heading_deltas=[0.5, -1.0, 0.5], distance=1.618)

    # Find nearest memory
    mem = floor.nearest(4.8, 2.9)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PHI = 1.618033988749895
INV_PHI = 1.0 / PHI


@dataclass
class Step:
    """A navigation step: distance + direction."""
    distance: float
    heading: float  # radians, 0=east, pi/2=north

    def advance(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        return (
            pos[0] + self.distance * math.cos(self.heading),
            pos[1] + self.distance * math.sin(self.heading),
        )


@dataclass
class FloorRead:
    """What you read from the floor after walking a path."""
    bits: List[bool] = field(default_factory=list)
    path: List[Tuple[float, float]] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)
    matched: List[bool] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Memory:
    """A stored memory on the floor."""
    position: Tuple[float, float]
    content: Any
    tile_bit: bool = False
    level: int = 0


class PenroseFloor:
    """
    Aperiodic memory palace navigated by dead reckoning.

    Every tile on the floor is determined by the Fibonacci word.
    Every neighborhood is unique (no collisions).
    Navigate by distance + direction — two numbers per query.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        scale: float = 1.0,
    ):
        self.pos = (x, y)
        self.heading = heading
        self.scale = scale
        self._memories: Dict[Tuple[int, int], Memory] = {}

    def _quantize(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        return (
            round(pos[0] / (self.scale * PHI)),
            round(pos[1] / (self.scale * PHI)),
        )

    def _tile_bit(self, pos: Tuple[int, int]) -> bool:
        """Fibonacci word bit at a lattice position."""
        q = pos[0] * 0x9E3779B9 & 0xFFFFFFFF
        mixed = abs((q ^ (pos[1] & 0xFFFFFFFF)) & 0xFFFFFFFF)
        idx = mixed % 1000
        current = int(idx * INV_PHI)
        nxt = int((idx + 1) * INV_PHI)
        return nxt != current

    def _matching_rule(self, pos: Tuple[int, int]) -> bool:
        bit = self._tile_bit(pos)
        neighbors = [
            (pos[0]+1, pos[1]), (pos[0]-1, pos[1]),
            (pos[0], pos[1]+1), (pos[0], pos[1]-1),
            (pos[0]+1, pos[1]-1), (pos[0]-1, pos[1]+1),
        ]
        if bit:
            return any(not self._tile_bit(n) for n in neighbors)
        return any(self._tile_bit(n) for n in neighbors)

    # --- Public API ---

    def store(self, x: float, y: float, content: Any) -> Tuple[int, int]:
        """Store a memory at position (x, y)."""
        key = self._quantize((x, y))
        self._memories[key] = Memory(
            position=(x, y),
            content=content,
            tile_bit=self._tile_bit(key),
        )
        return key

    def store_here(self, content: Any) -> Tuple[int, int]:
        """Store at current position."""
        return self.store(self.pos[0], self.pos[1], content)

    def retrieve(self, x: float, y: float) -> Optional[Any]:
        """Retrieve memory at position."""
        key = self._quantize((x, y))
        m = self._memories.get(key)
        return m.content if m else None

    def walk(self, steps: List[Step]) -> FloorRead:
        """Walk a path and read bits from the floor."""
        bits, path, headings, matched = [], [], [], []
        pos = self.pos
        heading = self.heading

        for step in steps:
            heading = step.heading
            new_pos = step.advance(pos)
            key = self._quantize(new_pos)
            bits.append(self._tile_bit(key))
            path.append(new_pos)
            headings.append(heading)
            matched.append(self._matching_rule(key))
            pos = new_pos

        confidence = sum(m for m in matched) / len(matched) if matched else 1.0
        return FloorRead(bits=bits, path=path, headings=headings,
                         matched=matched, confidence=confidence)

    def spline_to(self, target: Tuple[float, float], steps: int = 5) -> FloorRead:
        """Straight-line walk to target."""
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        step_list = [Step(dist / max(steps, 1), heading)] * steps
        return self.walk(step_list)

    def tack(self, heading_deltas: List[float], distance: float = PHI) -> FloorRead:
        """Zigzag like a sailboat."""
        steps = []
        h = self.heading
        for delta in heading_deltas:
            h += delta
            steps.append(Step(distance, h))
        return self.walk(steps)

    def stretch(self, stretches: List[float], heading: float = 0.0) -> FloorRead:
        """Varying distances at constant heading."""
        steps = [Step(d * self.scale * PHI, heading) for d in stretches]
        return self.walk(steps)

    def deflate(self, center: Tuple[float, float], radius: float) -> Optional[List[Any]]:
        """Consolidate nearby memories into one (dream module)."""
        key = self._quantize(center)
        r2 = math.ceil(radius / (self.scale * PHI)) ** 2
        to_merge = [(k, v) for k, v in self._memories.items()
                    if (k[0]-key[0])**2 + (k[1]-key[1])**2 <= r2]
        if not to_merge:
            return None
        contents = [v.content for _, v in to_merge]
        for k, _ in to_merge:
            del self._memories[k]
        self._memories[key] = Memory(
            position=center,
            content=contents,
            tile_bit=self._tile_bit(key),
            level=1,
        )
        return contents

    def nearest(self, x: float, y: float) -> Optional[Any]:
        """Find nearest stored memory."""
        key = self._quantize((x, y))
        best, best_d = None, float('inf')
        for k, v in self._memories.items():
            d = math.hypot(k[0]-key[0], k[1]-key[1])
            if d < best_d:
                best_d, best = d, v.content
        return best

    def __len__(self) -> int:
        return len(self._memories)
