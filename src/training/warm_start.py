"""
Warm-start / resume: manifest JSON next to checkpoints for inspection and resume_from_saved script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

WARM_START_MANIFEST_NAME = "warm_start_metadata.json"
TRAINING_LATEST_NAME = "training_latest.pt"


def _rel_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_safe_metrics_history(metrics_history: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics_history.items():
        if not isinstance(v, list):
            continue
        row: List[Any] = []
        for x in v:
            if isinstance(x, (float, np.floating)):
                row.append(float(x))
            elif isinstance(x, (int, np.integer)):
                row.append(int(x))
            else:
                row.append(x)
        out[k] = row
    return out


def write_warm_start_manifest(
    manifest_path: Path,
    *,
    checkpoint_path: Path,
    config_path: Optional[Path],
    mode: str,
    num_epochs_target: int,
    last_completed_epoch: int,
    metrics_history: Dict[str, List[float]],
    best_val_loss: float,
    best_val_mae: float,
    best_val_iou: float,
    loss_plot_path: Optional[Path],
    project_root: Path,
    config_snapshot: Optional[dict] = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "checkpoint_path": _rel_to_project(checkpoint_path, project_root),
        "config_path": _rel_to_project(config_path, project_root) if config_path else None,
        "mode": mode,
        "num_epochs_target": int(num_epochs_target),
        "last_completed_epoch": int(last_completed_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_mae": float(best_val_mae),
        "best_val_iou": float(best_val_iou),
        "metrics_history": _json_safe_metrics_history(metrics_history),
        "loss_plot_path": _rel_to_project(loss_plot_path, project_root) if loss_plot_path else None,
    }
    if config_snapshot is not None:
        data["config_snapshot"] = config_snapshot
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_warm_start_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def plot_loss_from_manifest_metrics(
    metrics_history: dict,
    output_path: Optional[Path] = None,
):
    """Build loss+LR figure from metrics_history (same style as training)."""
    from src.training.visualization import plot_loss_simple

    epochs = metrics_history.get("epochs") or []
    train_loss = metrics_history.get("train_loss") or []
    val_loss = metrics_history.get("val_loss") or []
    lr = metrics_history.get("learning_rate")
    if not epochs or not train_loss or not val_loss:
        return None
    fig = plot_loss_simple(
        [int(x) for x in epochs],
        [float(x) for x in train_loss],
        [float(x) for x in val_loss],
        learning_rate=[float(x) for x in lr] if lr else None,
        output_path=output_path,
    )
    return fig
