#!/usr/bin/env python3
"""
Run a sequence of training experiments defined by YAML override files.

Each experiment file lives in configs/experiments/ and contains overrides
on top of the base training_config.yaml. Results are logged to MLflow.

Usage:
    poetry run python scripts/run_experiment_sequence.py \
        --experiments exp_00_baseline.yaml exp_01_augmentation.yaml

    poetry run python scripts/run_experiment_sequence.py --all
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import yaml

from src.utils.path_utils import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_base_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_experiment_overrides(experiment_path: Path) -> dict:
    with open(experiment_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_experiment_overrides(base_config: dict, overrides: dict) -> dict:
    merged = _deep_merge(base_config, overrides)
    for key in ("experiment_name", "mode", "run_intention"):
        merged.pop(key, None)
    return merged


def _run_single_experiment(
    experiment_path: Path,
    base_config: dict,
    project_root: Path,
) -> dict:
    overrides = _load_experiment_overrides(experiment_path)
    experiment_name = overrides.get("experiment_name", experiment_path.stem)
    mode = overrides.get("mode", "production")
    run_intention = overrides.get("run_intention")
    config = _apply_experiment_overrides(base_config, overrides)

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info("Intention: %s", run_intention or "(not set)")
    logger.info("Config file: %s", experiment_path.name)
    logger.info("Mode: %s", mode)
    logger.info("=" * 60)

    from scripts.train_model import train_model_with_config

    start_time = time.time()
    try:
        best_val_loss = train_model_with_config(
            config=config,
            mode=mode,
            run_name=experiment_name,
            config_path=project_root / "configs" / "training_config.yaml",
            run_intention=run_intention,
        )
        elapsed = time.time() - start_time
        logger.info(
            "DONE: %s — best_val_loss=%.6f, elapsed=%.1f min",
            experiment_name, best_val_loss, elapsed / 60,
        )
        return {
            "experiment": experiment_name,
            "status": "success",
            "best_val_loss": best_val_loss,
            "elapsed_min": round(elapsed / 60, 1),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("FAILED: %s — %s (elapsed %.1f min)", experiment_name, e, elapsed / 60)
        return {
            "experiment": experiment_name,
            "status": "error",
            "error": str(e),
            "elapsed_min": round(elapsed / 60, 1),
        }


def _print_summary(results: list) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT SEQUENCE SUMMARY")
    print("=" * 70)
    for r in results:
        status = r["status"]
        name = r["experiment"]
        elapsed = r["elapsed_min"]
        if status == "success":
            print(f"  {name:40s}  loss={r['best_val_loss']:.6f}  ({elapsed:.0f} min)")
        else:
            print(f"  {name:40s}  FAILED: {r.get('error', '?')}  ({elapsed:.0f} min)")
    print("=" * 70)


def main():
    project_root = get_project_root(Path(__file__))
    experiments_dir = project_root / "configs" / "experiments"

    parser = argparse.ArgumentParser(description="Run experiment sequence")
    parser.add_argument(
        "--experiments", nargs="+", metavar="FILE",
        help="Experiment YAML filenames (relative to configs/experiments/)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all exp_*.yaml files in configs/experiments/ in sorted order",
    )
    args = parser.parse_args()

    if args.all:
        experiment_files = sorted(experiments_dir.glob("exp_*.yaml"))
    elif args.experiments:
        experiment_files = [experiments_dir / f for f in args.experiments]
    else:
        parser.error("Specify --experiments or --all")
        return

    for f in experiment_files:
        if not f.exists():
            logger.error("Experiment file not found: %s", f)
            sys.exit(1)

    base_config = _load_base_config(project_root)
    logger.info("Loaded base config. Running %d experiment(s).", len(experiment_files))

    results = []
    for exp_file in experiment_files:
        result = _run_single_experiment(exp_file, base_config, project_root)
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
