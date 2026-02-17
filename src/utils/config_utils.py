"""
Training config utilities: hyperparameter application from YAML or MLflow run.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_training_path_key(mode: str, tile_size: int) -> str:
    """Return the paths config key for (mode, tile_size). Synthetic uses synthetic_parenthesis_256 or _512."""
    if mode == "synthetic_parenthesis":
        return f"synthetic_parenthesis_{tile_size}"
    if tile_size == 256:
        return mode
    return f"{mode}_512"

import yaml

logger = logging.getLogger(__name__)

# Mapping from best_hyperparameters.yaml "hyperparameters" keys to config paths (tuple of keys)
BEST_HP_CONFIG_PATHS: Dict[str, tuple] = {
    "tile_size": ("data", "tile_size"),
    "learning_rate": ("training", "learning_rate"),
    "batch_size": ("training", "batch_size"),
    "weight_decay": ("training", "weight_decay"),
    "focal_alpha": ("training", "focal_alpha"),
    "focal_gamma": ("training", "focal_gamma"),
    "loss_function": ("training", "loss_function"),
    "encoder_name": ("model", "encoder", "name"),
    "decoder_dropout": ("model", "decoder_dropout"),
    "lr_scheduler_patience": ("training", "lr_scheduler_patience"),
    "lr_scheduler_factor": ("training", "lr_scheduler_factor"),
    "max_grad_norm": ("training", "max_grad_norm"),
    "unfreeze_after_epoch": ("model", "encoder", "unfreeze_after_epoch"),
}

# Keys to print when displaying applied hyperparameters (CLI feedback)
APPLIED_HP_DISPLAY_KEYS: List[str] = [
    "loss_function",
    "learning_rate",
    "batch_size",
    "encoder_name",
    "unfreeze_after_epoch",
    "focal_alpha",
    "focal_gamma",
    "decoder_dropout",
    "weight_decay",
    "max_grad_norm",
]


def parse_param_value(s: str) -> Any:
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(config: dict, path_keys: tuple, value: Any) -> None:
    d = config
    for k in path_keys[:-1]:
        d = d.setdefault(k, {})
    d[path_keys[-1]] = value


def apply_best_hyperparameters(config: Dict[str, Any], best_hp_path: Path) -> Dict[str, Any]:
    """
    Override config with values from best_hyperparameters.yaml (in-place).
    Returns loaded data for logging.
    """
    with open(best_hp_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    hp = data.get("hyperparameters") or {}
    for key, path_keys in BEST_HP_CONFIG_PATHS.items():
        if key not in hp:
            continue
        _set_nested(config, path_keys, hp[key])
    logger.info("Applied best hyperparameters from %s", best_hp_path)
    return data


def apply_hyperparameters_from_mlflow_run(
    config: Dict[str, Any],
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Override config with params from an MLflow run (in-place).
    Returns dict for logging (run_id, hyperparameters, etc.).
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params or {}
    hp: Dict[str, Any] = {}
    for key, path_keys in BEST_HP_CONFIG_PATHS.items():
        flat_key = ".".join(path_keys)
        raw = (
            params.get(f"best_hparams.{key}")
            or params.get(f"optuna.{key}")
            or params.get(flat_key)
        )
        if raw is None:
            continue
        hp[key] = parse_param_value(str(raw))
        _set_nested(config, path_keys, hp[key])
    best_val = None
    if "best_hparams.best_validation_loss" in params:
        try:
            best_val = float(params["best_hparams.best_validation_loss"])
        except (TypeError, ValueError):
            pass
    if best_val is None and run.data.metrics:
        best_val = run.data.metrics.get("best_val_loss")
    logger.info("Applied hyperparameters from MLflow run %s", run_id)
    return {
        "source": "mlflow_run",
        "run_id": run_id,
        "best_validation_loss": best_val,
        "best_trial_number": params.get("best_hparams.best_trial_number") or params.get("optuna_trial"),
        "hyperparameters": hp,
    }
