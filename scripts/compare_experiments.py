#!/usr/bin/env python3
"""
Compare experiment results from MLflow runs.

Reads MLflow tracking data and prints a side-by-side comparison table
of experiments matching a name pattern.

Usage:
    poetry run python scripts/compare_experiments.py --prefix exp_
    poetry run python scripts/compare_experiments.py --runs exp_00_baseline exp_01_augmentation
"""

import argparse
import logging
from pathlib import Path

import mlflow

from src.utils.path_utils import get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_TO_COMPARE = [
    "best_val_loss",
    "best_val_mae",
    "best_val_iou",
    "final_epoch",
]

PARAMS_TO_SHOW = [
    "loss_function",
    "learning_rate",
    "augmentation",
    "illumination_filter",
    "use_dem",
    "use_slope",
    "unfreeze_after_epoch",
    "bce_pos_weight",
    "num_epochs",
    "early_stopping_patience",
]


def _find_experiment_runs(experiment_name: str, run_names: list = None, prefix: str = None):
    mlflow_client = mlflow.tracking.MlflowClient()
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error("MLflow experiment '%s' not found", experiment_name)
        return []

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )

    if run_names:
        runs = [r for r in runs if r.info.run_name in run_names]
    elif prefix:
        runs = [r for r in runs if r.info.run_name and r.info.run_name.startswith(prefix)]

    return runs


def _format_value(value):
    if value is None:
        return "-"
    try:
        f = float(value)
        if abs(f) < 0.001:
            return f"{f:.6f}"
        return f"{f:.4f}"
    except (ValueError, TypeError):
        return str(value)


def _print_comparison_table(runs):
    if not runs:
        print("No matching runs found.")
        return

    names = [r.info.run_name or r.info.run_id[:8] for r in runs]
    col_width = max(20, max(len(n) for n in names) + 2)

    header = f"{'Metric/Param':<30s}" + "".join(f"{n:>{col_width}s}" for n in names)
    print("\n" + "=" * len(header))
    print("EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    print("  METRICS:")
    for metric_key in METRICS_TO_COMPARE:
        row = f"  {metric_key:<28s}"
        for run in runs:
            val = run.data.metrics.get(metric_key)
            row += f"{_format_value(val):>{col_width}s}"
        print(row)

    print()
    print("  PARAMS:")
    for param_key in PARAMS_TO_SHOW:
        values = [run.data.params.get(param_key) for run in runs]
        if all(v is None for v in values):
            continue
        row = f"  {param_key:<28s}"
        for val in values:
            row += f"{_format_value(val):>{col_width}s}"
        print(row)

    print()
    print("  STATUS:")
    row = f"  {'status':<28s}"
    for run in runs:
        row += f"{run.info.status:>{col_width}s}"
    print(row)

    row = f"  {'duration_min':<28s}"
    for run in runs:
        if run.info.end_time and run.info.start_time:
            dur = (run.info.end_time - run.info.start_time) / 60_000
            row += f"{dur:>{col_width}.1f}"
        else:
            row += f"{'-':>{col_width}s}"
    print(row)

    print("=" * len(header))


def main():
    project_root = get_project_root(Path(__file__))
    mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")

    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument("--prefix", type=str, default="exp_", help="Run name prefix to filter")
    parser.add_argument("--runs", nargs="+", help="Specific run names to compare")
    parser.add_argument(
        "--experiment-name", type=str, default="lobe_detection",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    runs = _find_experiment_runs(
        args.experiment_name,
        run_names=args.runs,
        prefix=args.prefix if not args.runs else None,
    )
    logger.info("Found %d matching runs", len(runs))
    _print_comparison_table(runs)


if __name__ == "__main__":
    main()
