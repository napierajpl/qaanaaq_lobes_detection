"""Helpers for reading MLflow run metrics/params and building loss-plot inputs."""

from pathlib import Path
from typing import Optional


def read_metric_by_step(metric_path: Path) -> list[tuple[int, float]]:
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


def read_param(params_dir: Path, key: str) -> Optional[str]:
    path = params_dir / key
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def format_param_display(val: Optional[str]) -> str:
    if val is None:
        return "?"
    try:
        f = float(val)
        s = f"{f:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except (ValueError, TypeError):
        return val


def build_config_summary(params_dir: Path) -> str:
    keys = [
        "model.architecture", "model.dropout", "model.encoder.name", "model.encoder.pretrained",
        "model.encoder.freeze_encoder", "model.encoder.unfreeze_after_epoch", "model.decoder_dropout",
        "model.output_activation", "training.batch_size", "training.num_epochs", "training.learning_rate",
        "training.loss_function", "data.tile_size", "data.train_split",
    ]
    numeric_keys = {"dropout", "decoder_dropout", "batch_size", "num_epochs", "learning_rate", "tile_size", "train_split"}
    parts = []
    for k in keys:
        v = read_param(params_dir, k)
        short = k.split(".")[-1] if "." in k else k
        if short in numeric_keys and v is not None:
            v = format_param_display(v)
        parts.append(f"{short}={v}" if v is not None else f"{short}=?")
    return " | ".join(parts)


def early_stop_counter_from_val_loss(val_loss: list[float], min_delta: float = 1e-5) -> list[int]:
    counter = []
    best = float("inf")
    for v in val_loss:
        if v < best - min_delta:
            best = v
            counter.append(0)
        else:
            counter.append(counter[-1] + 1 if counter else 1)
    return counter
