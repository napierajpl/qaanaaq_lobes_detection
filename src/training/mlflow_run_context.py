"""
MLflow run setup: run start message, config/trial logging, initial loss placeholder, loss plot options.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow

from src.utils.mlflow_utils import log_training_config, get_intention_suggestion
from src.utils.proximity_utils import infer_proximity_token, detect_proximity_params

logger = logging.getLogger(__name__)


def _fmt_config_value(v: Any) -> str:
    if v is None:
        return "?"
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def build_loss_plot_options(
    config: dict,
    num_train_tiles: int,
    num_val_tiles: int,
    early_stopping_patience: Optional[int],
    early_stopping_min_delta: float,
) -> Dict[str, Any]:
    enc = config.get("model", {}).get("encoder", {})
    tr = config.get("training", {})
    da = config.get("data", {})
    target_mode = (config.get("target_mode") or "proximity").lower()
    out_act = "sigmoid" if target_mode == "binary" else config.get("model", {}).get("output_activation", "?")
    parts = [
        f"arch={config.get('model', {}).get('architecture', '?')}",
        f"drop={_fmt_config_value(config.get('model', {}).get('dropout'))}",
        f"enc={enc.get('name', '?')}",
        f"pretrain={enc.get('pretrained', '?')}",
        f"freeze={enc.get('freeze_encoder', '?')}",
        f"unfreeze_ep={enc.get('unfreeze_after_epoch', '?')}",
        f"dec_drop={_fmt_config_value(config.get('model', {}).get('decoder_dropout'))}",
        f"out_act={out_act}",
        f"batch={tr.get('batch_size', '?')}",
        f"ep={tr.get('num_epochs', '?')}",
        f"lr={_fmt_config_value(tr.get('learning_rate'))}",
        f"loss={tr.get('loss_function', '?')}",
        f"tile={da.get('tile_size', '?')}",
        f"train_split={_fmt_config_value(da.get('train_split'))}",
    ]
    return {
        "config_summary": " | ".join(parts),
        "num_train_tiles": num_train_tiles,
        "num_val_tiles": num_val_tiles,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta if early_stopping_patience else None,
        "freeze_encoder": enc.get("freeze_encoder"),
        "unfreeze_after_epoch": enc.get("unfreeze_after_epoch"),
    }


def prompt_run_intention(trial: Optional[Any]) -> Optional[str]:
    """If trial is None, optionally prompt for run intention and set MLflow tag. Return intention or None."""
    if trial is not None:
        return None
    active = mlflow.active_run()
    if active is not None:
        suggestion = get_intention_suggestion(active.info.experiment_id, active.info.run_id)
        if suggestion:
            logger.info("Suggestion: %s", suggestion)
    try:
        run_intention = input("Run intention (subtitle): ").strip() or None
    except EOFError:
        run_intention = None
    if active is not None and run_intention:
        mlflow.set_tag("run_intention", run_intention)
    return run_intention


def get_loss_plot_path_and_print_run_start(run_name: str) -> Optional[Path]:
    """Call inside mlflow.start_run(). Logs run start, returns loss plot path if file tracking."""
    active = mlflow.active_run()
    if active is None:
        return None
    exp_id = active.info.experiment_id
    run_id = active.info.run_id
    tracking_uri = mlflow.get_tracking_uri()
    logger.info(
        "MLFLOW START experiment_id=%s run_id=%s run_name=%s",
        exp_id, run_id, run_name,
    )
    loss_plot_path = None
    if tracking_uri.startswith("file:"):
        base = tracking_uri.replace("file:", "").rstrip("/")
        plots_dir = Path(base) / str(exp_id) / run_id / "artifacts" / "plots"
        loss_plot_path = plots_dir / "loss.png"
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Plots (updated each epoch): %s", loss_plot_path)
    logger.info("tracking_uri=%s", tracking_uri)
    return loss_plot_path


def log_run_config_and_trial_metadata(
    config: dict,
    mode: str,
    run_name: str,
    trial: Optional[Any],
    applied_best_hparams: Optional[dict],
    num_params: int,
    trainable_params: int,
    num_train_tiles: int,
    num_val_tiles: int,
    train_subsample_ratio: float,
    architecture: str,
    in_channels: int,
    iou_threshold: float,
    filtered_tiles_path: Path,
    targets_dir: Path,
    val_tiles: List[dict],
) -> None:
    log_training_config(config)
    mlflow.log_param("mode", mode)
    mlflow.log_param("target_mode", config.get("target_mode", "proximity"))
    mlflow.log_param("num_params", num_params)
    if applied_best_hparams is not None:
        hp = applied_best_hparams.get("hyperparameters") or {}
        if applied_best_hparams.get("source") == "mlflow_run":
            run_id = applied_best_hparams.get("run_id")
            best_val = applied_best_hparams.get("best_validation_loss")
            mlflow.set_tag("hp_source_run_id", str(run_id))
            if best_val is not None:
                mlflow.log_param("hp_from_run.best_validation_loss", best_val)
            for k, v in hp.items():
                mlflow.log_param(f"hp_from_run.{k}", v)
            logger.info(
                "Using hyperparameters from MLflow run_id=%s val_loss=%s loss=%s lr=%s batch=%s encoder=%s unfreeze_epoch=%s",
                run_id, best_val, hp.get("loss_function"), hp.get("learning_rate"), hp.get("batch_size"),
                hp.get("encoder_name"), hp.get("unfreeze_after_epoch"),
            )
        else:
            trial_num = applied_best_hparams.get("best_trial_number")
            best_val = applied_best_hparams.get("best_validation_loss")
            mlflow.set_tag("best_hparams_trial", str(trial_num))
            mlflow.log_param("best_hparams.best_validation_loss", best_val)
            mlflow.log_param("best_hparams.best_trial_number", trial_num)
            for k, v in hp.items():
                mlflow.log_param(f"best_hparams.{k}", v)
            logger.info(
                "Using best hyperparameters: trial=%s val_loss=%.4f loss=%s lr=%.2e batch=%s encoder=%s unfreeze_epoch=%s",
                trial_num, best_val,
                hp.get("loss_function"), hp.get("learning_rate"), hp.get("batch_size"),
                hp.get("encoder_name"), hp.get("unfreeze_after_epoch"),
            )
    mlflow.log_param("trainable_params", trainable_params)
    mlflow.log_param("num_train_tiles", num_train_tiles)
    mlflow.log_param("num_val_tiles", num_val_tiles)
    if train_subsample_ratio < 1.0:
        mlflow.log_param("data.train_subsample_ratio", train_subsample_ratio)

    if trial is not None:
        mlflow.set_tag("optuna_trial", str(trial.number))
        mlflow.set_tag("optuna_study", "lobe_detection_hp_tuning")
        for key, value in trial.params.items():
            mlflow.log_param(f"optuna.{key}", value)

    active = mlflow.active_run()
    if trial is not None and active is not None and hasattr(trial, "set_user_attr"):
        try:
            trial.set_user_attr("mlflow_experiment_id", active.info.experiment_id)
            trial.set_user_attr("mlflow_run_id", active.info.run_id)
            trial.set_user_attr("mlflow_run_name", run_name)
            trial.set_user_attr("mlflow_tracking_uri", mlflow.get_tracking_uri())
            trial.set_user_attr("mode", mode)
            trial.set_user_attr("model_architecture", architecture)
            trial.set_user_attr("model_in_channels", in_channels)
            trial.set_user_attr("model_out_channels", int(config["model"].get("out_channels", 1)))
            trial.set_user_attr("training_iou_threshold", float(iou_threshold))
            trial.set_user_attr("filtered_tiles_path", str(filtered_tiles_path))
            trial.set_user_attr("targets_dir", str(targets_dir))
            trial.set_user_attr("proximity_token", infer_proximity_token(str(targets_dir)))
        except Exception:
            logger.debug("Failed to set trial user attributes", exc_info=True)

    mlflow.log_param("model.architecture", architecture)
    if architecture == "satlaspretrain_unet":
        enc_cfg = config["model"].get("encoder", {})
        mlflow.log_param("model.encoder_name", enc_cfg.get("name", "unknown"))
        mlflow.log_param("model.encoder_pretrained", enc_cfg.get("pretrained", False))
        mlflow.log_param("model.encoder_frozen", enc_cfg.get("freeze_encoder", False))
        mlflow.log_param("model.encoder_unfreeze_epoch", enc_cfg.get("unfreeze_after_epoch", 0))

    proximity_max_value, proximity_max_distance = detect_proximity_params(
        targets_dir, val_tiles, sample_size=5
    )
    if proximity_max_value is not None:
        mlflow.log_param("data.proximity_max_value", proximity_max_value)
        mlflow.log_param("data.proximity_max_distance", proximity_max_distance)
        logger.info(
            "Detected proximity map: max_value=%s, max_distance=%s",
            proximity_max_value, proximity_max_distance,
        )


def create_initial_loss_placeholder() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training started – plot will update after each epoch.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    mlflow.log_figure(fig, "plots/loss.png")
    plt.close(fig)
