#!/usr/bin/env python3
"""
Generate board_config.json from physical board measurements.

Measurements needed:
  - marker_size_m: physical size of each ArUco marker (side length)
  - center_to_center_m: distance between marker centers (horizontal and vertical)
  - dictionary: ArUco dictionary name

Marker layout (looking at the board from the front):

    +Y (up)
     |
  1  |  3       (top-left, top-right)
     |
─────┼─────  +X (right)
     |
  4  |  2       (bottom-left, bottom-right)
     |

Board center = (0, 0)
"""

import json
import math

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURE YOUR BOARD MEASUREMENTS HERE
# ─────────────────────────────────────────────────────────────────────────────

# ArUco dictionary used for the detection board
DICTIONARY = "DICT_6X6_250"

# Physical size of each ArUco marker in meters
MARKER_SIZE_M = 0.0225  # 22.5mm

# Distance between marker CENTERS (horizontal and vertical) in meters
CENTER_TO_CENTER_M = 0.115  # 11.5cm

# Marker IDs and their positions
# Format: { id: position }
# Positions: "top_left", "top_right", "bottom_left", "bottom_right"
MARKER_POSITIONS = {
    1: "top_left",
    3: "top_right",
    4: "bottom_left",
    2: "bottom_right",
}

# Rotation of each marker in degrees (0 if not rotated)
MARKER_ROTATIONS = {
    1: 0,
    3: 0,
    4: 0,
    2: 0,
}

# Output file path
OUTPUT_FILE = "board_config.json"

# ─────────────────────────────────────────────────────────────────────────────
#  CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_top_left_xy(position: str, marker_size: float, center_to_center: float):
    """
    Compute top-left corner (x, y) of a marker given its position label.

    Board frame convention:
      +X = right
      +Y = up
      Origin = board center

    The top-left corner of a marker = marker_center - (marker_size/2) in X,
                                       marker_center + (marker_size/2) in Y
    """
    half_c2c = center_to_center / 2.0
    half_size = marker_size / 2.0

    # Marker center positions
    centers = {
        "top_left":     (-half_c2c, +half_c2c),
        "top_right":    (+half_c2c, +half_c2c),
        "bottom_left":  (-half_c2c, -half_c2c),
        "bottom_right": (+half_c2c, -half_c2c),
    }

    if position not in centers:
        raise ValueError(f"Unknown position: {position}. "
                         f"Must be one of {list(centers.keys())}")

    cx, cy = centers[position]

    # Top-left corner = center shifted by half marker size
    top_left_x = cx - half_size
    top_left_y = cy + half_size

    return top_left_x, top_left_y


def main():
    print("=" * 60)
    print("  Board Config Generator")
    print("=" * 60)
    print()
    print(f"  Dictionary:          {DICTIONARY}")
    print(f"  Marker size:         {MARKER_SIZE_M*1000:.1f} mm")
    print(f"  Center-to-center:    {CENTER_TO_CENTER_M*1000:.1f} mm")
    print()

    markers = {}

    for marker_id, position in MARKER_POSITIONS.items():
        x, y = compute_top_left_xy(position, MARKER_SIZE_M, CENTER_TO_CENTER_M)
        rotation = MARKER_ROTATIONS.get(marker_id, 0)

        markers[str(marker_id)] = {
            "top_left_xy_m": [round(x, 6), round(y, 6)],
            "rotation_deg": rotation,
        }

        print(f"  Marker {marker_id} ({position:14s}): "
              f"top_left=({x:+.6f}, {y:+.6f})  rotation={rotation}°")

    print()

    # Estimate board size from outermost marker corners
    board_size = CENTER_TO_CENTER_M + MARKER_SIZE_M

    config = {
        "board_size_m": round(board_size, 6),
        "marker_size_m": MARKER_SIZE_M,
        "dictionary": DICTIONARY,
        "frame_description": (
            "Board frame origin at board center. "
            "+X points right on the board. "
            "+Y points up on the board. "
            "Z=0 is the board plane."
        ),
        "markers": markers,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved to: {OUTPUT_FILE}")
    print()
    print("  Generated config:")
    print("  " + "-" * 56)
    print(json.dumps(config, indent=2))
    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()