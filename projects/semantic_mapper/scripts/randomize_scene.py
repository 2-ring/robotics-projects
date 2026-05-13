#!/usr/bin/env python3
"""Randomize object positions on tables for Q1 scene.

Each table has 4 corners (inset from edge); the corner closest to the
nearest room corner is excluded, giving 3 valid slots per table = 9 total.
We randomly pick 5 of the 9 slots and assign the 5 objects.

Usage:
  python3 randomize_scene.py <input.xml> <output.xml> [--seed N]
"""

import argparse
import random
import re
from pathlib import Path

OBJECTS = ["mug", "bottle", "box", "bowl", "can"]
OBJ_Z = 0.35  # table-top z

# Margin inset from table edge for object placement
_MARGIN = 0.06

# Table definitions: (center_x, center_y, half_x, half_y, room_corner_x, room_corner_y)
_TABLES = [
    (0.85, 0.85, 0.22, 0.22, 1.5, 1.5),     # table_1 NE
    (-0.85, -0.85, 0.25, 0.20, -1.5, -1.5),  # table_2 SW
    (-0.85, 0.85, 0.20, 0.22, -1.5, 1.5),    # table_3 NW
]


def _get_valid_slots():
    """Return the 9 valid table-corner positions (3 per table)."""
    slots = []
    for cx, cy, hx, hy, rcx, rcy in _TABLES:
        # 4 corners inset by margin
        corners = [
            (cx - hx + _MARGIN, cy - hy + _MARGIN),
            (cx + hx - _MARGIN, cy - hy + _MARGIN),
            (cx - hx + _MARGIN, cy + hy - _MARGIN),
            (cx + hx - _MARGIN, cy + hy - _MARGIN),
        ]
        # Exclude the corner closest to the room corner
        corners.sort(key=lambda c: (c[0] - rcx) ** 2 + (c[1] - rcy) ** 2)
        # corners[0] is closest to room corner → exclude it
        slots.extend(corners[1:])
    return slots


# Pre-computed for import by other modules
VALID_SLOTS = _get_valid_slots()


def randomize(input_xml: str, seed: int | None = None) -> tuple[str, dict]:
    """Return (new XML, placements dict) with randomized object positions."""
    rng = random.Random(seed)

    slots = list(VALID_SLOTS)
    rng.shuffle(slots)
    chosen = slots[:5]

    objs = list(OBJECTS)
    rng.shuffle(objs)

    placements = {}
    for obj, (sx, sy) in zip(objs, chosen):
        placements[obj] = (sx, sy, OBJ_Z)

    output = input_xml
    for obj_name, (px, py, pz) in placements.items():
        pattern = rf'(<body\s+name="obj_{obj_name}"\s+pos=")[^"]*(")'
        replacement = rf'\g<1>{px:.3f} {py:.3f} {pz:.3f}\2'
        output = re.sub(pattern, replacement, output)

    return output, placements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    xml_in = Path(args.input).read_text()
    xml_out, placements = randomize(xml_in, args.seed)
    Path(args.output).write_text(xml_out)

    for obj in OBJECTS:
        if obj in placements:
            p = placements[obj]
            print(f"  {obj}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")


if __name__ == "__main__":
    main()
