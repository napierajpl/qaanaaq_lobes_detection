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
        print(f"  Each layer: {tile_dir}/<layer_name>/")
        print(f"  Targets: {tile_dir}/targets/")
        print(f"  Filtered list: {tile_dir}/filtered_tiles.json")
        if self.dev_mode:
            print("\nNote: All processing done on 1024x1024 cropped files for quick testing")
        print("\nFiltering: Empty RGB tiles excluded. Background-only tiles excluded.")
        print("\n" + "="*60)
        print("Timing Summary")
        print("="*60)
        for description, elapsed in self.step_times:
            print(f"  {description:50s} {elapsed:7.2f}s")
        print(f"\n  {'Total time':50s} {total_time:7.2f}s")
        print("="*60)


def _derived_layer_steps(
    cfg: Dict[str, Any],
    mode_key: str,
) -> List[Tuple[str, List[str]]]:
    derived = cfg.get("derived_layers", {})
    steps: List[Tuple[str, List[str]]] = []
    for name, layer_cfg in derived.items():
        mode_cfg = layer_cfg.get(mode_key)
        if mode_cfg is None:
            continue
        transform = layer_cfg["transform"]
        params = layer_cfg.get("params", {})
        inputs = mode_cfg["inputs"]
        output = mode_cfg["output"]
        cmd: List[str] = [
            "poetry", "run", "python", "scripts/create_derived_layer.py",
            "--transform", transform,
            "--output", output,
        ]
        for input_name, input_path in inputs.items():
            cmd.extend(["--input", f"{input_name}={input_path}"])
        for param_name, param_val in params.items():
            cmd.extend(["--param", f"{param_name}={param_val}"])
        desc = f"  Computing derived layer: {name} (transform={transform})"
        steps.append((desc, cmd))
    return steps


def _tiling_steps(
    cfg: Dict[str, Any],
    mode_key: str,
    tile_size: int,
    tile_dir: str,
) -> List[Tuple[str, List[str]]]:
    tile_sources = cfg.get("tile_sources", {}).get(mode_key, {})
    overlap = str(cfg.get("tiling", {}).get("overlap", 0.3))
    steps: List[Tuple[str, List[str]]] = []
    for layer_name, raster_path in tile_sources.items():
        desc = f"  Tiling {layer_name} ({tile_size}x{tile_size})"
        cmd = [
            "poetry", "run", "python", "scripts/create_tiles.py",
            "-i", raster_path,
            "-o", f"{tile_dir}/{layer_name}",
            "--tile-size", str(tile_size),
            "--overlap", overlap,
            "--no-organize",
        ]
        steps.append((desc, cmd))
    return steps


def _filter_step(
    cfg: Dict[str, Any],
    tile_dir: str,
) -> Tuple[str, List[str]]:
    filt = cfg.get("filtering", {})
    lobe_threshold = str(filt.get("lobe_threshold", 5.0))
    return (
        "Filtering tiles (excluding empty RGB + background-only)",
        [
            "poetry", "run", "python", "scripts/filter_tiles.py",
            "--features", f"{tile_dir}/rgb",
            "--targets", f"{tile_dir}/targets",
            "--output", f"{tile_dir}/filtered_tiles.json",
            "--exclude-background",
            "--lobe-threshold", lobe_threshold,
        ],
    )


def production_steps(tile_size: int) -> List[Tuple[str, List[str]]]:
    cfg = _load_pipeline_config()
    raw = cfg["raw"]
    prod = cfg["production"]
    prox = cfg["proximity"]

    steps: List[Tuple[str, List[str]]] = [
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
    ]

    steps.extend(_derived_layer_steps(cfg, "production"))

    tile_dir = tile_dir_for_pipeline(False, tile_size)
    steps.extend(_tiling_steps(cfg, "production", tile_size, tile_dir))
    steps.append(_filter_step(cfg, tile_dir))
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

    crop_size = f"{crop['width']}x{crop['height']}"
    steps: List[Tuple[str, List[str]]] = [
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
            f"Step 3: Generating proximity map for cropped lobes ({prox['max_distance']}px)",
            [
                "poetry", "run", "python", "scripts/generate_proximity_map.py",
                "-i", dev["lobes_cropped"],
                "-o", dev["proximity_map"],
                "--max-value", str(prox["max_value"]),
                "--max-distance", str(prox["max_distance"]),
            ],
        ),
    ]

    steps.extend(_derived_layer_steps(cfg, "dev"))

    tile_dir = tile_dir_for_pipeline(True, tile_size)
    steps.extend(_tiling_steps(cfg, "dev", tile_size, tile_dir))
    steps.append(_filter_step(cfg, tile_dir))
    return steps
