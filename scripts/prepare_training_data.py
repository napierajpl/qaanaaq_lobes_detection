#!/usr/bin/env python3
"""
Prepare training data: create VRT stack, generate proximity map, and tile.

This script orchestrates the full training data preparation pipeline.
"""

import sys
from pathlib import Path

from src.utils.path_utils import get_project_root
from src.data_processing.prepare_training_steps import (
    PipelineRunner,
    production_steps,
    dev_steps,
)


def main():
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Prepare training data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run dev pipeline (cropped 1024x1024 files)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Tile size in pixels (256 or 512). Default: 256.",
    )

    args = parser.parse_args()

    runner = PipelineRunner(project_root, dev_mode=args.dev, tile_size=args.tile_size)
    steps = dev_steps(args.tile_size) if args.dev else production_steps(args.tile_size)

    if args.dev:
        print("\n" + "="*60)
        print("Training Data Preparation Pipeline (DEV - Cropped Files)")
        print("Working with 1024x1024 cropped files for quick testing")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Training Data Preparation Pipeline")
        print("="*60)

    try:
        runner.run_steps(steps)
        runner.print_summary()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
