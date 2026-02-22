#!/usr/bin/env python3
"""
Regenerate enhanced loss.png for MLflow runs from given dates (e.g. today and yesterday).
Excludes a given run_id if specified. Overwrites artifacts/plots/loss.png.
Usage: poetry run python scripts/regenerate_loss_plots_for_runs.py [--exclude-run-id ID] [--days 2]
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Regenerate loss plots for recent MLflow runs")
    parser.add_argument("--mlruns", type=Path, default=Path("mlruns"), help="Path to mlruns dir")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="586083506121040615",
        help="Experiment ID",
    )
    parser.add_argument(
        "--exclude-run-id",
        type=str,
        default="61e3c008cb5b4efe811bda648c902546",
        help="Run ID to skip",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Include runs from today and the last (days-1) calendar days (default: 2 = today and yesterday)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print run IDs, do not regenerate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    mlruns = project_root / args.mlruns if not args.mlruns.is_absolute() else args.mlruns
    exp_dir = mlruns / args.experiment_id
    if not exp_dir.is_dir():
        print(f"Experiment dir not found: {exp_dir}", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    today = now.date()
    allowed_dates = {today - timedelta(days=i) for i in range(args.days)}
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and (d / "meta.yaml").exists()]
    run_ids_to_process = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        if run_id == args.exclude_run_id:
            continue
        meta_path = run_dir / "meta.yaml"
        text = meta_path.read_text(encoding="utf-8")
        start_time_ms = None
        for line in text.splitlines():
            if line.startswith("start_time:"):
                val = line.split(":", 1)[1].strip()
                if val and val != "null":
                    start_time_ms = int(val)
                break
        if start_time_ms is None:
            continue
        dt = datetime.fromtimestamp(start_time_ms / 1000.0, tz=timezone.utc)
        if dt.date() in allowed_dates:
            run_ids_to_process.append((run_id, dt))

    run_ids_to_process.sort(key=lambda x: x[1], reverse=True)

    if not run_ids_to_process:
        print("No runs found in the last {} day(s).".format(args.days))
        return 0

    print("Regenerating loss.png for {} run(s) (excluding {}):".format(
        len(run_ids_to_process), args.exclude_run_id))
    for run_id, dt in run_ids_to_process:
        print("  {}  {}".format(run_id, dt.strftime("%Y-%m-%d %H:%M UTC")))

    if args.dry_run:
        return 0

    script = project_root / "scripts" / "plot_loss_from_mlflow_run.py"
    for run_id, _ in run_ids_to_process:
        out_path = exp_dir / run_id / "artifacts" / "plots" / "loss.png"
        if not (exp_dir / run_id / "metrics" / "train_loss").exists():
            print("Skipping {} (no train_loss metrics)".format(run_id), file=sys.stderr)
            continue
        cmd = [
            sys.executable,
            str(script),
            "--run-id", run_id,
            "--mlruns", str(mlruns),
            "--experiment-id", args.experiment_id,
            "--output", str(out_path),
        ]
        ret = subprocess.run(cmd, cwd=project_root)
        if ret.returncode != 0:
            print("Failed: {}".format(run_id), file=sys.stderr)
            return ret.returncode
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
