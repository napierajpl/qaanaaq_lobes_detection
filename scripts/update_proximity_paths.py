#!/usr/bin/env python3
"""Update proximity map paths in filtered_tiles.json from 10px to 20px."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path


def update_proximity_paths(filtered_tiles_path: Path, old_pattern: str, new_pattern: str) -> None:
    """
    Update proximity map paths in filtered tiles JSON.

    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        old_pattern: Old path pattern to replace (e.g., "proximity10px")
        new_pattern: New path pattern (e.g., "proximity20px")
    """
    with open(filtered_tiles_path, 'r') as f:
        data = json.load(f)

    updated_count = 0
    for tile in data.get("tiles", []):
        if "targets_path" in tile:
            old_path = tile["targets_path"]
            if old_pattern in old_path:
                new_path = old_path.replace(old_pattern, new_pattern)
                tile["targets_path"] = new_path
                updated_count += 1

    # Save updated JSON
    with open(filtered_tiles_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated {updated_count} tile paths from '{old_pattern}' to '{new_pattern}'")
    print(f"Updated file: {filtered_tiles_path}")


def main():
    """Update proximity paths in filtered tiles."""
    import argparse

    project_root = get_project_root(__file__)

    parser = argparse.ArgumentParser(
        description="Update proximity map paths in filtered_tiles.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/processed/tiles/train/filtered_tiles.json",
        help="Path to filtered_tiles.json (default: data/processed/tiles/train/filtered_tiles.json)",
    )
    parser.add_argument(
        "--old-pattern",
        type=str,
        default="proximity10px",
        help="Old path pattern to replace (default: proximity10px)",
    )
    parser.add_argument(
        "--new-pattern",
        type=str,
        default="proximity20px",
        help="New path pattern (default: proximity20px)",
    )

    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    update_proximity_paths(input_path, args.old_pattern, args.new_pattern)


if __name__ == "__main__":
    main()
