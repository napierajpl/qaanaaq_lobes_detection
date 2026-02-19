"""
MLflow utilities for experiment tracking.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import torch.nn as nn

# Short display names for config keys in intention suggestion
_PARAM_DISPLAY_NAMES = {
    "learning_rate": "lr",
    "loss_function": "loss",
    "num_epochs": "ep",
    "batch_size": "batch",
    "freeze_encoder": "freeze",
    "unfreeze_after_epoch": "unfreeze_ep",
    "train_split": "train_split",
    "tile_size": "tile",
    "output_activation": "out_act",
    "architecture": "arch",
}


def _sanitize_run_id_part(value: str) -> str:
    """Keep only alphanumeric and underscores for use in run_id (path-safe)."""
    if not value:
        return "unknown"
    s = str(value).lower().strip()
    s = re.sub(r"[^a-z0-9_\-]", "_", s)
    return s or "unknown"


def build_user_friendly_run_id(config: Dict[str, Any], trial: Optional[Any] = None) -> str:
    """
    Build a readable run_id from timestamp, architecture and loss (e.g. 2026-02-18-19-23-45_satlaspretrain_unet_bce).
    Optional trial appends _t003 for Optuna runs so each trial has a unique id.
    """
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    arch = _sanitize_run_id_part(config.get("model", {}).get("architecture", "unet"))
    loss = _sanitize_run_id_part(config.get("training", {}).get("loss_function", "unknown"))
    parts = [ts, arch, loss]
    if trial is not None and getattr(trial, "number", None) is not None:
        parts.append(f"t{trial.number:03d}")
    return "_".join(parts)


def setup_mlflow_experiment(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    """
    Setup MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI (default: file:./mlruns)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration to MLflow.

    Args:
        config: Configuration dictionary
    """
    # Flatten nested config for MLflow
    params: Dict[str, str] = {}

    def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                flatten_dict(value, prefix=f"{prefix}{key}.")
            else:
                params[f"{prefix}{key}"] = str(value)

    flatten_dict(config)
    mlflow.log_params(params)


def _short_param_key(key: str) -> str:
    short = key.split(".")[-1] if "." in key else key
    return _PARAM_DISPLAY_NAMES.get(short, short)


def get_intention_suggestion(experiment_id: str, current_run_id: str) -> str:
    """
    Compare current run params with the most recent other run in the experiment.
    Returns a one-line suggestion of what changed (e.g. "lr 1e-6→1e-7, loss mse→bce").
    """
    client = mlflow.MlflowClient()
    try:
        current_run = client.get_run(current_run_id)
        current_params = current_run.data.params or {}
    except Exception:
        return ""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.end_time DESC"],
        max_results=20,
    )
    prev_run_id = None
    for r in runs:
        if r.info.run_id != current_run_id and r.info.end_time is not None:
            prev_run_id = r.info.run_id
            break
    if not prev_run_id:
        return "No previous run to compare."

    try:
        prev_run = client.get_run(prev_run_id)
        prev_params = prev_run.data.params or {}
    except Exception:
        return ""

    diffs: List[str] = []
    all_keys = sorted(set(current_params.keys()) | set(prev_params.keys()))
    for key in all_keys:
        cur = current_params.get(key)
        prev = prev_params.get(key)
        if cur is None or prev is None or cur != prev:
            if cur != prev:
                display = _short_param_key(key)
                diffs.append(f"{display} {prev}->{cur}")
    if not diffs:
        return "No changes vs last run."
    return "Changes vs last run: " + ", ".join(diffs)


def _get_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory in bytes.

    Args:
        directory: Path to directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.is_file():
                    total_size += filepath.stat().st_size
    except (OSError, PermissionError):
        pass
    return total_size


def save_model(model: nn.Module, artifact_path: str = "model") -> float:
    """
    Save PyTorch model to MLflow and log its size.

    Args:
        model: PyTorch model
        artifact_path: Path within artifacts to save model

    Returns:
        Model size in megabytes
    """
    # Save model to MLflow
    mlflow.pytorch.log_model(model, artifact_path)

    # Calculate model size on disk
    # MLflow saves artifacts to: mlruns/{experiment_id}/{run_id}/artifacts/{artifact_path}
    run = mlflow.active_run()
    if run:
        tracking_uri = mlflow.get_tracking_uri()

        # Handle file:// URI
        if tracking_uri.startswith("file://"):
            base_path = Path(tracking_uri.replace("file://", ""))
        else:
            base_path = Path(tracking_uri)

        # Construct artifact path
        artifact_dir = base_path / str(run.info.experiment_id) / run.info.run_id / "artifacts" / artifact_path

        if artifact_dir.exists():
            size_bytes = _get_directory_size(artifact_dir)
            size_mb = size_bytes / (1024 * 1024)

            # Log model size as parameter
            mlflow.log_param("model_size_mb", f"{size_mb:.2f}")
            mlflow.log_param("model_size_bytes", str(size_bytes))

            return size_mb

    return 0.0


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step/epoch number
    """
    mlflow.log_metrics(metrics, step=step)
