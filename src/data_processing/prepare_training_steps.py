import subprocess
import time
from pathlib import Path
from typing import List, Tuple

from src.utils.path_utils import tile_dir_for_pipeline


class PipelineRunner:
    def __init__(self, project_root: Path, dev_mode: bool = False, tile_size: int = 256):
        self.project_root = project_root
        self.dev_mode = dev_mode
        self.tile_size = tile_size
        self.step_times: List[Tuple[str, float]] = []

    def run_command(self, cmd: List[str], description: str) -> None:
        print(f"\n{'='*60}")
        print(description)
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

    def run_steps(self, steps: List[Tuple[str, List[str]]]) -> None:
        for description, cmd in steps:
            self.run_command(cmd, description)

    def print_summary(self) -> None:
        total_time = sum(t for _, t in self.step_times)
        tile_dir = tile_dir_for_pipeline(self.dev_mode, self.tile_size)
        print("\n" + "="*60)
        print("Pipeline Complete")
        print("="*60)
        print(f"\nTiles created in: {tile_dir}/ ({self.tile_size}x{self.tile_size})")
        print(f"  Features: {tile_dir}/features/")
        print(f"  Targets: {tile_dir}/targets/")
        print(f"  Filtered list: {tile_dir}/filtered_tiles.json")
        if self.dev_mode:
            print("\nNote: All processing done on 1024x1024 cropped files for quick testing")
            print("Note: Feature VRT file: data/processed/raster/features_combined.vrt")
        print("\nFiltering: Empty RGB tiles excluded. Background-only tiles excluded.")
        print("\n" + "="*60)
        print("Timing Summary")
        print("="*60)
        for description, elapsed in self.step_times:
            print(f"  {description:50s} {elapsed:7.2f}s")
        print(f"\n  {'Total time':50s} {total_time:7.2f}s")
        print("="*60)


def production_steps(tile_size: int) -> List[Tuple[str, List[str]]]:
    steps = [
        (
            "Step 1: Generating proximity map for lobes (20px)",
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", "data/processed/raster/rasterized_lobes_raw_by_code.tif",
                "-o", "data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif",
                "--max-value", "20",
                "--max-distance", "20",
            ],
        ),
        (
            "  Resampling DEM to match RGB...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", "data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif",
                "-r", "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
                "-o", "data/processed/raster/dem_from_arcticDEM_resampled.tif",
            ],
        ),
        (
            "  Resampling slope to match RGB...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", "data/raw/raster/dem/slope_from_dem_cropped.tif",
                "-r", "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
                "-o", "data/processed/raster/slope_from_dem_resampled.tif",
            ],
        ),
        (
            "Step 3: Creating VRT stack for feature layers (RGB + DEM + Slope)",
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
                "data/processed/raster/dem_from_arcticDEM_resampled.tif",
                "data/processed/raster/slope_from_dem_resampled.tif",
                "-o", "data/processed/raster/features_combined.vrt",
            ],
        ),
    ]
    tile_dir = tile_dir_for_pipeline(False, tile_size)
    steps.extend([
        (
            f"Step 4: Creating tiles for features ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/features_combined.vrt",
                "-o", f"{tile_dir}/features",
                "--tile-size", str(tile_size),
                "--overlap", "0.3",
            ],
        ),
        (
            f"Step 5: Creating tiles for targets (proximity map 20px, {tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif",
                "-o", f"{tile_dir}/targets",
                "--tile-size", str(tile_size),
                "--overlap", "0.3",
            ],
        ),
        (
            "Step 6: Filtering tiles and computing baselines (excluding empty RGB tiles)",
            [
                "poetry", "run", "python", "scripts/filter_tiles.py",
                "--features", f"{tile_dir}/features",
                "--targets", f"{tile_dir}/targets",
                "--output", f"{tile_dir}/filtered_tiles.json",
                "--exclude-background",
                "--lobe-threshold", "5.0",
            ],
        ),
    ])
    return steps


def dev_steps(tile_size: int) -> List[Tuple[str, List[str]]]:
    steps = [
        (
            "Step 1: Cropping input files to 1024x1024",
            [
                "poetry", "run", "python", "scripts/crop_raster.py",
                "-i", "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
                "--lon", "-69.2674970", "--lat", "77.4766436",
                "--width", "1024", "--height", "1024",
                "-o", "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif",
            ],
        ),
        (
            "  Cropping DEM...",
            [
                "poetry", "run", "python", "scripts/crop_raster.py",
                "-i", "data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif",
                "--lon", "-69.2674970", "--lat", "77.4766436",
                "--width", "1024", "--height", "1024",
                "-o", "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024.tif",
            ],
        ),
        (
            "  Cropping slope...",
            [
                "poetry", "run", "python", "scripts/crop_raster.py",
                "-i", "data/raw/raster/dem/slope_from_dem_cropped.tif",
                "--lon", "-69.2674970", "--lat", "77.4766436",
                "--width", "1024", "--height", "1024",
                "-o", "data/processed/raster/dev/slope_from_dem_cropped1024x1024.tif",
            ],
        ),
        (
            "  Cropping lobes raster...",
            [
                "poetry", "run", "python", "scripts/crop_raster.py",
                "-i", "data/processed/raster/rasterized_lobes_raw_by_code.tif",
                "--lon", "-69.2674970", "--lat", "77.4766436",
                "--width", "1024", "--height", "1024",
                "-o", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024.tif",
            ],
        ),
        (
            "Step 2: Resampling DEM and slope to match RGB resolution",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024.tif",
                "-r", "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif",
                "-o", "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif",
            ],
        ),
        (
            "  Resampling slope...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", "data/processed/raster/dev/slope_from_dem_cropped1024x1024.tif",
                "-r", "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif",
                "-o", "data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif",
            ],
        ),
        (
            "Step 3: Generating proximity map for cropped lobes (20px)",
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024.tif",
                "-o", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024_proximity20px.tif",
                "--max-value", "20",
                "--max-distance", "20",
            ],
        ),
        (
            "Step 4: Creating VRT stack for cropped feature layers (RGB + DEM + Slope)",
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                "data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif",
                "data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif",
                "data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif",
                "-o", "data/processed/raster/dev/features_combined_cropped1024x1024.vrt",
            ],
        ),
    ]
    tile_dir = tile_dir_for_pipeline(True, tile_size)
    steps.extend([
        (
            f"Step 5: Creating tiles for features ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/dev/features_combined_cropped1024x1024.vrt",
                "-o", f"{tile_dir}/features",
                "--tile-size", str(tile_size),
                "--overlap", "0.3",
            ],
        ),
        (
            f"Step 6: Creating tiles for targets (proximity map 20px, {tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", "data/processed/raster/dev/rasterized_lobes_raw_by_code_cropped1024x1024_proximity20px.tif",
                "-o", f"{tile_dir}/targets",
                "--tile-size", str(tile_size),
                "--overlap", "0.3",
            ],
        ),
        (
            "Step 7: Filtering tiles and computing baselines (excluding empty RGB tiles)",
            [
                "poetry", "run", "python", "scripts/filter_tiles.py",
                "--features", f"{tile_dir}/features",
                "--targets", f"{tile_dir}/targets",
                "--output", f"{tile_dir}/filtered_tiles.json",
                "--exclude-background",
                "--lobe-threshold", "5.0",
            ],
        ),
    ])
    return steps
