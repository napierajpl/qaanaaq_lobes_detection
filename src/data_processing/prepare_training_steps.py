import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.utils.path_utils import tile_dir_for_pipeline

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_pipeline_paths.yaml"


def _load_pipeline_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
        print(f"  Slope-stripes: {tile_dir}/slope_stripes_channel/")
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
    cfg = _load_pipeline_config()
    raw = cfg["raw"]
    prod = cfg["production"]
    prox = cfg["proximity"]
    gabor = cfg["gabor"]
    tiling = cfg["tiling"]
    filt = cfg["filtering"]

    steps = [
        (
            f"Step 1: Generating proximity map for lobes ({prox['max_distance']}px)",
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", cfg["processed"]["lobes_raster"],
                "-o", prod["proximity_map"],
                "--max-value", str(prox["max_value"]),
                "--max-distance", str(prox["max_distance"]),
            ],
        ),
        (
            "  Resampling DEM to match RGB...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", raw["dem"],
                "-r", raw["rgb"],
                "-o", prod["dem_resampled"],
            ],
        ),
        (
            "  Resampling slope to match RGB...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", raw["slope"],
                "-r", raw["rgb"],
                "-o", prod["slope_resampled"],
            ],
        ),
        (
            f"  Generating slope-stripes channel (RGB + DEM, Gabor "
            f"freq={gabor['frequency']} sigma={gabor['sigma']} "
            f"align={gabor['alignment_power']})...",
            [
                "poetry", "run", "python", "scripts/create_slope_stripes_channel.py",
                "--method", gabor["method"],
                "--gabor-frequency", str(gabor["frequency"]),
                "--gabor-sigma", str(gabor["sigma"]),
                "--alignment-power", str(gabor["alignment_power"]),
                "-i", raw["rgb"],
                "-d", prod["dem_resampled"],
                "-o", prod["slope_stripes"],
            ],
        ),
        (
            "Step 3: Creating VRT stack for feature layers (RGB + DEM + Slope)",
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                raw["rgb"],
                prod["dem_resampled"],
                prod["slope_resampled"],
                "-o", prod["features_vrt"],
            ],
        ),
    ]
    tile_dir = tile_dir_for_pipeline(False, tile_size)
    overlap = str(tiling["overlap"])
    steps.extend([
        (
            f"Step 4: Creating tiles for features ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", prod["features_vrt"],
                "-o", f"{tile_dir}/features",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
            ],
        ),
        (
            f"Step 5: Creating tiles for targets "
            f"(proximity map {prox['max_distance']}px, {tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", prod["proximity_map"],
                "-o", f"{tile_dir}/targets",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
            ],
        ),
        (
            f"Step 5b: Creating tiles for slope-stripes channel ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", prod["slope_stripes"],
                "-o", f"{tile_dir}/slope_stripes_channel",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
                "--no-organize",
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
                "--lobe-threshold", str(filt["lobe_threshold"]),
            ],
        ),
    ])
    return steps


def _crop_step(description: str, raw_path: str, output_path: str,
               crop: Dict[str, Any]) -> Tuple[str, List[str]]:
    return (
        description,
        [
            "poetry", "run", "python", "scripts/crop_raster.py",
            "-i", raw_path,
            "--lon", str(crop["lon"]), "--lat", str(crop["lat"]),
            "--width", str(crop["width"]), "--height", str(crop["height"]),
            "-o", output_path,
        ],
    )


def dev_steps(tile_size: int) -> List[Tuple[str, List[str]]]:
    cfg = _load_pipeline_config()
    raw = cfg["raw"]
    dev = cfg["dev"]
    crop = cfg["dev_crop"]
    prox = cfg["proximity"]
    gabor = cfg["gabor"]
    tiling = cfg["tiling"]
    filt = cfg["filtering"]

    crop_size = f"{crop['width']}x{crop['height']}"
    steps = [
        _crop_step(f"Step 1: Cropping input files to {crop_size}",
                   raw["rgb"], dev["rgb_cropped"], crop),
        _crop_step("  Cropping DEM...",
                   raw["dem"], dev["dem_cropped"], crop),
        _crop_step("  Cropping slope...",
                   raw["slope"], dev["slope_cropped"], crop),
        _crop_step("  Cropping lobes raster...",
                   cfg["processed"]["lobes_raster"], dev["lobes_cropped"], crop),
        (
            "Step 2: Resampling DEM and slope to match RGB resolution",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", dev["dem_cropped"],
                "-r", dev["rgb_cropped"],
                "-o", dev["dem_resampled"],
            ],
        ),
        (
            "  Resampling slope...",
            [
                "poetry", "run", "python", "scripts/resample_raster.py",
                "-i", dev["slope_cropped"],
                "-r", dev["rgb_cropped"],
                "-o", dev["slope_resampled"],
            ],
        ),
        (
            f"  Generating slope-stripes channel (cropped RGB + DEM, Gabor "
            f"freq={gabor['frequency']} sigma={gabor['sigma']} "
            f"align={gabor['alignment_power']})...",
            [
                "poetry", "run", "python", "scripts/create_slope_stripes_channel.py",
                "--method", gabor["method"],
                "--gabor-frequency", str(gabor["frequency"]),
                "--gabor-sigma", str(gabor["sigma"]),
                "--alignment-power", str(gabor["alignment_power"]),
                "-i", dev["rgb_cropped"],
                "-d", dev["dem_resampled"],
                "-o", dev["slope_stripes"],
            ],
        ),
        (
            f"Step 3: Generating proximity map for cropped lobes ({prox['max_distance']}px)",
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", dev["lobes_cropped"],
                "-o", dev["proximity_map"],
                "--max-value", str(prox["max_value"]),
                "--max-distance", str(prox["max_distance"]),
            ],
        ),
        (
            "Step 4: Creating VRT stack for cropped feature layers (RGB + DEM + Slope)",
            [
                "poetry", "run", "python", "scripts/create_vrt_stack.py",
                "-i",
                dev["rgb_cropped"],
                dev["dem_resampled"],
                dev["slope_resampled"],
                "-o", dev["features_vrt"],
            ],
        ),
    ]
    tile_dir = tile_dir_for_pipeline(True, tile_size)
    overlap = str(tiling["overlap"])
    steps.extend([
        (
            f"Step 5: Creating tiles for features ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", dev["features_vrt"],
                "-o", f"{tile_dir}/features",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
            ],
        ),
        (
            f"Step 6: Creating tiles for targets "
            f"(proximity map {prox['max_distance']}px, {tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", dev["proximity_map"],
                "-o", f"{tile_dir}/targets",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
            ],
        ),
        (
            f"Step 6b: Creating tiles for slope-stripes channel ({tile_size}x{tile_size})",
            [
                "poetry", "run", "python", "scripts/create_tiles.py",
                "-i", dev["slope_stripes"],
                "-o", f"{tile_dir}/slope_stripes_channel",
                "--tile-size", str(tile_size),
                "--overlap", overlap,
                "--no-organize",
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
                "--lobe-threshold", str(filt["lobe_threshold"]),
            ],
        ),
    ])
    return steps
