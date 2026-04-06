#!/usr/bin/env python3
"""Compare two MLflow runs."""

from pathlib import Path


def read_mlflow_value(file_path: Path) -> str:
    """Read a single value from MLflow param/metric file."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_run_data(run_id: str, experiment_id: str = "586083506121040615") -> dict:
    """Get all parameters and final metrics for a run."""
    base_path = Path("mlruns") / experiment_id / run_id

    data = {
        "run_id": run_id,
        "params": {},
        "metrics": {},
    }

    # Read parameters
    params_dir = base_path / "params"
    if params_dir.exists():
        for param_file in params_dir.iterdir():
            param_name = param_file.name
            param_value = read_mlflow_value(param_file)
            data["params"][param_name] = param_value

    # Read metrics (get last value for each)
    metrics_dir = base_path / "metrics"
    if metrics_dir.exists():
        for metric_file in metrics_dir.iterdir():
            metric_name = metric_file.name
            # MLflow metrics files have format: timestamp value
            # We want the last (most recent) value
            if metric_file.exists():
                with open(metric_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        # Last line has the final value
                        last_line = lines[-1].strip()
                        parts = last_line.split()
                        if len(parts) >= 2:
                            try:
                                value = float(parts[1])
                                data["metrics"][metric_name] = value
                            except ValueError:
                                pass

    return data

def compare_runs(run_id1: str, run_id2: str):
    """Compare two MLflow runs."""
    data1 = get_run_data(run_id1)
    data2 = get_run_data(run_id2)

    print("=" * 80)
    print("COMPARING RUNS")
    print("=" * 80)
    print(f"\nRun 1: {run_id1}")
    print(f"Run 2: {run_id2}\n")

    # Key parameters to compare
    key_params = [
        "data.proximity_max_value",
        "data.proximity_max_distance",
        "training.loss_function",
        "training.focal_alpha",
        "training.focal_gamma",
        "training.learning_rate",
        "training.batch_size",
        "num_train_tiles",
    ]

    print("\n" + "=" * 80)
    print("KEY PARAMETERS")
    print("=" * 80)
    print(f"{'Parameter':<40} {'Run 1':<20} {'Run 2':<20}")
    print("-" * 80)

    for param in key_params:
        val1 = data1["params"].get(param, "N/A")
        val2 = data2["params"].get(param, "N/A")
        marker = " [DIFF]" if val1 != val2 else ""
        print(f"{param:<40} {str(val1):<20} {str(val2):<20}{marker}")

    # Key metrics to compare
    key_metrics = [
        "val_iou",
        "val_mae",
        "val_rmse",
        "val_loss",
        "train_loss",
        "val_baseline_mae",
        "val_improvement_over_baseline",
    ]

    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"{'Metric':<40} {'Run 1':<20} {'Run 2':<20} {'Change':<20}")
    print("-" * 80)

    for metric in key_metrics:
        val1 = data1["metrics"].get(metric)
        val2 = data2["metrics"].get(metric)

        if val1 is None or val2 is None:
            print(f"{metric:<40} {str(val1) if val1 is not None else 'N/A':<20} {str(val2) if val2 is not None else 'N/A':<20}")
        else:
            change = val2 - val1
            change_pct = (change / val1 * 100) if val1 != 0 else 0
            if metric in ["val_iou", "val_improvement_over_baseline"]:
                marker = " [BETTER]" if change > 0 else " [WORSE]" if change < 0 else ""
            else:
                marker = " [BETTER]" if change < 0 else " [WORSE]" if change > 0 else ""
            print(f"{metric:<40} {val1:<20.6f} {val2:<20.6f} {change:+.6f} ({change_pct:+.1f}%){marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    iou1 = data1["metrics"].get("val_iou", 0)
    iou2 = data2["metrics"].get("val_iou", 0)
    mae1 = data1["metrics"].get("val_mae", float('inf'))
    mae2 = data2["metrics"].get("val_mae", float('inf'))
    baseline1 = data1["metrics"].get("val_baseline_mae", 0)
    baseline2 = data2["metrics"].get("val_baseline_mae", 0)

    print(f"\nIoU: {iou1:.6f} -> {iou2:.6f} ({iou2-iou1:+.6f}, {(iou2-iou1)/iou1*100 if iou1 > 0 else 0:+.1f}%)")
    print(f"MAE: {mae1:.6f} -> {mae2:.6f} ({mae2-mae1:+.6f}, {(mae2-mae1)/mae1*100 if mae1 > 0 else 0:+.1f}%)")
    print(f"Baseline MAE: {baseline1:.6f} -> {baseline2:.6f}")

    if baseline1 > 0 and baseline2 > 0:
        ratio1 = mae1 / baseline1
        ratio2 = mae2 / baseline2
        print(f"Model/Baseline ratio: {ratio1:.2f}x -> {ratio2:.2f}x")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two MLflow runs")
    parser.add_argument("run_id1", help="First run ID")
    parser.add_argument("run_id2", help="Second run ID")

    args = parser.parse_args()

    compare_runs(args.run_id1, args.run_id2)
