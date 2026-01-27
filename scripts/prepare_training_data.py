#!/usr/bin/env python3
"""
Prepare training data: create VRT stack, generate proximity map, and tile.

This script orchestrates the full training data preparation pipeline.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root
from tqdm import tqdm


class PipelineRunner:
    """Runs training data preparation pipeline steps."""

    def __init__(self, project_root: Path, dev_mode: bool = False):
        self.project_root = project_root
        self.dev_mode = dev_mode
        self.step_times = []

    def run_command(self, cmd: List[str], description: str) -> None:
        """Run a command with progress indication."""
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=False,
                text=True,
            )
            elapsed = time.time() - start_time
            self.step_times.append((description, elapsed))
            print(f"\n✓ Completed in {elapsed:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed with exit code {e.returncode}")
            raise

    def prepare_production_data(self) -> None:
        """Run production training data preparation pipeline."""
        print("\n" + "="*60)
        print("Training Data Preparation Pipeline")
        print("="*60)

        # Step 1: Generate proximity map (20px)
        self.run_command(
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", "data/processed/raster/rasterized_lobes_raw_by_code.tif",
                "-o", "data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif",
                "--max-value", "20",
                "--max-distance", "20",
            ],
            "Step 1: Generating proximity map for lobes (20px)"
        )

        # Step 2: Resample DEM and slope to match RGB resolution
        print("\n" + "="*60)
        print("Step 2: Resampling DEM and slope to match RGB resolution")
        print("="*60)

        reference_raster = "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"

        resamples = [
            ("DEM", "data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif",
             "data/processed/raster/dem_from_arcticDEM_resampled.tif"),
            ("slope", "data/raw/raster/dem/slope_from_dem_cropped.tif",
             "data/processed/raster/slope_from_dem_resampled.tif"),
        ]

        for name, input_path, output_path in tqdm(resamples, desc="Resampling"):
            self.run_command(
                [
                    "poetry", "run", "python", "scripts/resample_raster.py",
                    "-i", input_path,
                    "-r", reference_raster,
                    "-o", output_path,
                ],
                f"  Resampling {name}..."
            )

        # Step 3: Create VRT stack
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
                "data/processed/raster/dem_from_arcticDEM_resampled.tif",
                "data/processed/raster/slope_from_dem_resampled.tif",
                "-o", "data/processed/raster/features_combined.vrt",
            ],
            "Step 3: Creating VRT stack for feature layers (RGB + DEM + Slope)"
        )

        # Step 4: Create tiles for features
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/features_combined.vrt",
                "-o", "data/processed/tiles/train/features",
                "--tile-size", "256",
                "--overlap", "0.3",
            ],
            "Step 4: Creating tiles for features"
        )

        # Step 5: Create tiles for targets
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif",
                "-o", "data/processed/tiles/train/targets",
                "--tile-size", "256",
                "--overlap", "0.3",
            ],
            "Step 5: Creating tiles for targets (proximity map 20px)"
        )

        # Step 6: Filter tiles (with baseline computation)
        self.run_command(
            [
                "poetry", "run", "python", "scripts/filter_tiles.py",
                "--features", "data/processed/tiles/train/features",
                "--targets", "data/processed/tiles/train/targets",
                "--output", "data/processed/tiles/train/filtered_tiles.json",
                "--exclude-background",
                "--lobe-threshold", "5.0",  # For baseline computation
            ],
            "Step 6: Filtering tiles and computing baselines (excluding empty RGB tiles)"
        )

    def prepare_dev_data(self) -> None:
        """Run dev training data preparation pipeline (cropped files)."""
        print("\n" + "="*60)
        print("Training Data Preparation Pipeline (DEV - Cropped Files)")
        print("Working with 1024x1024 cropped files for quick testing")
        print("="*60)

        crop_lon = "-69.2674970"
        crop_lat = "77.4766436"
        crop_size = "1024"

        # Step 1: Crop all input files
        print("\n" + "="*60)
        print("Step 1: Cropping input files to 1024x1024")
        print("="*60)

        crops = [
            ("RGB imagery", "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
             "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"),
            ("DEM", "data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif",
             "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024.tif"),
            ("slope", "data/raw/raster/dem/slope_from_dem_cropped.tif",
             "data/processed/raster/dev/slope_from_dem_cropped1024x1024.tif"),
            ("lobes raster", "data/processed/raster/rasterized_lobes_raw_by_code.tif",
             "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024.tif"),
        ]

        for name, input_path, output_path in tqdm(crops, desc="Cropping files"):
            self.run_command(
                [
                    "poetry", "run", "python", "scripts/crop_raster.py",
                    "-i", input_path,
                    "--lon", crop_lon,
                    "--lat", crop_lat,
                    "--width", crop_size,
                    "--height", crop_size,
                    "-o", output_path,
                ],
                f"  Cropping {name}..."
            )

        # Step 2: Resample DEM and slope
        print("\n" + "="*60)
        print("Step 2: Resampling DEM and slope to match RGB resolution")
        print("="*60)

        resamples = [
            ("DEM", "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024.tif",
             "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif"),
            ("slope", "data/processed/raster/dev/slope_from_dem_cropped1024x1024.tif",
             "data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif"),
        ]

        reference = "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"

        for name, input_path, output_path in tqdm(resamples, desc="Resampling"):
            self.run_command(
                [
                    "poetry", "run", "python", "scripts/resample_raster.py",
                    "-i", input_path,
                    "-r", reference,
                    "-o", output_path,
                ],
                f"  Resampling {name}..."
            )

        # Step 3: Generate proximity map (20px)
        self.run_command(
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024.tif",
                "-o", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024_proximity20px.tif",
                "--max-value", "20",
                "--max-distance", "20",
            ],
            "Step 3: Generating proximity map for cropped lobes (20px)"
        )

        # Step 4: Create VRT stack
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif",
                "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif",
                "data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif",
                "-o", "data/processed/raster/dev/features_combined_cropped1024x1024.vrt",
            ],
            "Step 4: Creating VRT stack for cropped feature layers (RGB + DEM + Slope)"
        )

        # Step 5: Create tiles for features
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/dev/features_combined_cropped1024x1024.vrt",
                "-o", "data/processed/tiles/dev/train/features",
                "--tile-size", "256",
                "--overlap", "0.3",
            ],
            "Step 5: Creating tiles for features"
        )

        # Step 6: Create tiles for targets
        self.run_command(
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024_proximity20px.tif",
                "-o", "data/processed/tiles/dev/train/targets",
                "--tile-size", "256",
                "--overlap", "0.3",
            ],
            "Step 6: Creating tiles for targets (proximity map 20px)"
        )

        # Step 7: Filter tiles (with baseline computation)
        self.run_command(
            [
                "poetry", "run", "python", "scripts/filter_tiles.py",
                "--features", "data/processed/tiles/dev/train/features",
                "--targets", "data/processed/tiles/dev/train/targets",
                "--output", "data/processed/tiles/dev/train/filtered_tiles.json",
                "--exclude-background",
                "--lobe-threshold", "5.0",  # For baseline computation
            ],
            "Step 7: Filtering tiles and computing baselines (excluding empty RGB tiles)"
        )

    def print_summary(self) -> None:
        """Print pipeline summary with timing information."""
        total_time = sum(time for _, time in self.step_times)

        print("\n" + "="*60)
        print("Pipeline Complete")
        print("="*60)

        if self.dev_mode:
            print("\nTiles created in: data/processed/tiles/dev/train/")
            print("  Features: data/processed/tiles/dev/train/features/")
            print("  Targets: data/processed/tiles/dev/train/targets/")
            print("  Filtered list: data/processed/tiles/dev/train/filtered_tiles.json")
            print("\nNote: All processing done on 1024x1024 cropped files for quick testing")
        else:
            print("\nTiles created in: data/processed/tiles/train/")
            print("  Features: data/processed/tiles/train/features/")
            print("  Targets: data/processed/tiles/train/targets/")
            print("  Filtered list: data/processed/tiles/train/filtered_tiles.json")
            print("\nNote: Feature VRT file: data/processed/raster/features_combined.vrt")
            print("      (Virtual file - references original rasters, saves disk space)")

        print("\nFiltering: Empty RGB tiles excluded. Background-only tiles excluded.")
        print("           Adjust filtering options in filter_tiles.py if needed.")

        print("\n" + "="*60)
        print("Timing Summary")
        print("="*60)
        for description, elapsed in self.step_times:
            print(f"  {description:50s} {elapsed:7.2f}s")
        print(f"\n  {'Total time':50s} {total_time:7.2f}s")
        print("="*60)


def main():
    """Main entry point."""
    import argparse

    project_root = get_project_root(__file__)

    parser = argparse.ArgumentParser(
        description="Prepare training data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run dev pipeline (cropped 1024x1024 files)",
    )

    args = parser.parse_args()

    runner = PipelineRunner(project_root, dev_mode=args.dev)

    try:
        if args.dev:
            runner.prepare_dev_data()
        else:
            runner.prepare_production_data()

        runner.print_summary()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
