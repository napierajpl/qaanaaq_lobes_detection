#!/usr/bin/env python3
"""
Filter training tiles based on data quality.

Excludes tiles with empty RGB data and optionally filters background-only tiles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.tile_filter import TileFilter
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import resolve_path, get_project_root


def main():
    project_root = get_project_root(__file__)

    parser = BaseCLIParser(
        description="Filter training tiles based on RGB data quality and target presence",
        project_root=project_root,
    )

    parser.set_epilog("""
Examples:
  # Filter tiles, exclude empty RGB and background-only tiles
  python scripts/filter_tiles.py \\
      --features data/processed/tiles/dev/train/features \\
      --targets data/processed/tiles/dev/train/targets \\
      --output data/processed/tiles/dev/train/filtered_tiles.json \\
      --exclude-background

  # Filter tiles, keep background-only tiles (default)
  python scripts/filter_tiles.py \\
      --features data/processed/tiles/dev/train/features \\
      --targets data/processed/tiles/dev/train/targets \\
      --output data/processed/tiles/dev/train/filtered_tiles.json

  # Filter with minimum target coverage threshold
  python scripts/filter_tiles.py \\
      --features data/processed/tiles/dev/train/features \\
      --targets data/processed/tiles/dev/train/targets \\
      --output data/processed/tiles/dev/train/filtered_tiles.json \\
      --min-target-coverage 0.01
    """)

    parser.parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Directory containing feature tiles (RGB)",
    )

    parser.parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Directory containing target tiles (proximity maps)",
    )

    parser.parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path for filtered tile list",
    )

    parser.parser.add_argument(
        "--min-rgb-coverage",
        type=float,
        default=0.01,
        help="Minimum fraction of RGB pixels that must be valid (default: 0.01 = 1%%)",
    )

    parser.parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude tiles with no targets (all background)",
    )

    parser.parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=None,
        help="Minimum fraction of target pixels that must be positive (optional)",
    )

    parser.parser.add_argument(
        "--lobe-threshold",
        type=float,
        default=5.0,
        help="Threshold for lobe pixels in baseline computation (default: 5.0)",
    )

    parser.parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip computing per-tile baseline metrics (faster but less informative)",
    )

    args = parser.parse_args()

    # Resolve paths
    features_dir = resolve_path(args.features, project_root)
    targets_dir = resolve_path(args.targets, project_root)
    output_file = resolve_path(args.output, project_root)

    # Validate directories exist
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    if not targets_dir.exists():
        raise FileNotFoundError(f"Targets directory not found: {targets_dir}")

    # Create filter
    tile_filter = TileFilter(
        min_rgb_coverage=args.min_rgb_coverage,
        include_background_only=not args.exclude_background,
        min_target_coverage=args.min_target_coverage,
    )

    # Filter tiles
    print("Filtering tiles...")
    print(f"  Features: {features_dir}")
    print(f"  Targets: {targets_dir}")
    print(f"  Min RGB coverage: {args.min_rgb_coverage}")
    print(f"  Include background-only: {not args.exclude_background}")
    if args.min_target_coverage:
        print(f"  Min target coverage: {args.min_target_coverage}")
    print()

    valid_tiles, stats_summary = tile_filter.filter_tile_pairs(
        features_dir=features_dir,
        targets_dir=targets_dir,
        output_file=output_file,
        compute_baselines=not args.no_baselines,
        lobe_threshold=args.lobe_threshold,
    )

    # Print summary
    tile_filter.print_summary(stats_summary)

    print(f"Filtered tile list saved to: {output_file}")
    print("Use this file in training scripts to load only valid tiles.")


if __name__ == "__main__":
    main()
