"""
MLflow utilities for experiment tracking.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch.nn as nn


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
