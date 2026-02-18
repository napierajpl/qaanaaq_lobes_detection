#!/usr/bin/env python3
"""
Generate the enhanced loss plot from an existing MLflow run (metrics + params).
Usage: poetry run python scripts/plot_loss_from_mlflow_run.py --run-id 52fd23bee17440cb979060fe7edeef65 [--output path]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.visualization import plot_loss


def _read_metric_by_step(metric_path: Path) -> list[tuple[int, float]]:
    out = []
    with open(metric_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                step = int(parts[2])
                value = float(parts[1])
                out.append((step, value))
    out.sort(key=lambda x: x[0])
    return out


def _read_param(params_dir: Path, key: str) -> str | None:
    path = params_dir / key
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _format_param_display(val: str | None) -> str:
    if val is None:
        return "?"
    try:
        f = float(val)
        s = f"{f:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except (ValueError, TypeError):
        return val


def _build_config_summary(params_dir: Path) -> str:
    keys = [
        "model.architecture",
        "model.dropout",
        "model.encoder.name",
        "model.encoder.pretrained",
        "model.encoder.freeze_encoder",
        "model.encoder.unfreeze_after_epoch",
        "model.decoder_dropout",
        "model.output_activation",
        "training.batch_size",
        "training.num_epochs",
        "training.learning_rate",
        "training.loss_function",
        "data.tile_size",
        "data.train_split",
    ]
    numeric_keys = {"dropout", "decoder_dropout", "batch_size", "num_epochs", "learning_rate", "tile_size", "train_split"}
    parts = []
    for k in keys:
        v = _read_param(params_dir, k)
        short = k.split(".")[-1] if "." in k else k
        if short in numeric_keys and v is not None:
            v = _format_param_display(v)
        parts.append(f"{short}={v}" if v is not None else f"{short}=?")
    return " | ".join(parts)


def _early_stop_counter_from_val_loss(val_loss: list[float], min_delta: float = 1e-5) -> list[int]:
    counter = []
    best = float("inf")
    for v in val_loss:
        if v < best - min_delta:
            best = v
            counter.append(0)
        else:
            counter.append(counter[-1] + 1 if counter else 1)
    return counter


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced loss plot from MLflow run")
    parser.add_argument("--run-id", required=True, help="MLflow run UUID (e.g. 52fd23bee17440cb979060fe7edeef65)")
    parser.add_argument(
        "--mlruns",
        type=Path,
        default=Path("mlruns"),
        help="Path to mlruns directory (default: mlruns)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="586083506121040615",
        help="Experiment ID (default: 586083506121040615)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for loss.png (default: run artifacts/plots/loss_regenerated.png)",
    )
    parser.add_argument(
        "--intention",
        type=str,
        default=None,
        help="Run intention (subtitle and info box, e.g. 'changed lr to 1e-7, sigmoid')",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    mlruns = project_root / args.mlruns if not args.mlruns.is_absolute() else args.mlruns
    run_dir = mlruns / args.experiment_id / args.run_id
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    metrics_dir = run_dir / "metrics"
    params_dir = run_dir / "params"
    train_steps = _read_metric_by_step(metrics_dir / "train_loss")
    val_steps = _read_metric_by_step(metrics_dir / "val_loss")
    if not train_steps or not val_steps:
        print("Missing train_loss or val_loss metrics.", file=sys.stderr)
        sys.exit(1)

    steps = [s for s, _ in train_steps]
    train_loss = [v for _, v in train_steps]
    val_loss = [v for _, v in val_steps]
    if len(train_loss) != len(val_loss):
        n = min(len(train_loss), len(val_loss))
        steps, train_loss, val_loss = steps[:n], train_loss[:n], val_loss[:n]

    patience_str = _read_param(params_dir, "training.early_stopping_patience")
    early_stopping_patience = int(patience_str) if patience_str and patience_str.isdigit() else None
    min_delta_str = _read_param(params_dir, "training.early_stopping_min_delta")
    min_delta = float(min_delta_str) if min_delta_str else 1e-5
    early_stop_counter = _early_stop_counter_from_val_loss(val_loss, min_delta) if early_stopping_patience else None

    num_train = _read_param(params_dir, "num_train_tiles")
    num_val = _read_param(params_dir, "num_val_tiles")
    num_train_tiles = int(num_train) if num_train and num_train.isdigit() else None
    num_val_tiles = int(num_val) if num_val and num_val.isdigit() else None
    config_summary = _build_config_summary(params_dir)

    freeze_str = _read_param(params_dir, "model.encoder.freeze_encoder")
    freeze_encoder = freeze_str and freeze_str.lower() in ("true", "1") if freeze_str is not None else None
    unfreeze_str = _read_param(params_dir, "model.encoder.unfreeze_after_epoch")
    unfreeze_after_epoch = int(unfreeze_str) if unfreeze_str and unfreeze_str.isdigit() else None

    fig = plot_loss(
        steps,
        train_loss,
        val_loss,
        min_val_loss=min(val_loss) if val_loss else None,
        early_stop_counter=early_stop_counter,
        early_stopping_patience=early_stopping_patience,
        config_summary=config_summary,
        num_train_tiles=num_train_tiles,
        num_val_tiles=num_val_tiles,
        freeze_encoder=freeze_encoder,
        unfreeze_after_epoch=unfreeze_after_epoch,
        run_intention=args.intention,
    )

    out_path = args.output
    if out_path is None:
        artifacts = run_dir / "artifacts" / "plots"
        artifacts.mkdir(parents=True, exist_ok=True)
        out_path = artifacts / "loss_regenerated.png"
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
