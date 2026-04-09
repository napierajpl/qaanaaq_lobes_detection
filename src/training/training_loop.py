"""Training loop: one-epoch train/validate, Optuna pruning, early stopping, best-model save, loss plot."""

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from src.training.layer_registry import LayerRegistry
from src.training.trainer import train_one_epoch, validate, save_training_checkpoint, ValidationResult
from src.training.visualization import (
    plot_loss_simple,
    show_best_predicted_tile,
    show_highest_iou_tile,
    create_prediction_tile_figures,
    create_representative_tiles_channel_figures,
)
from src.training.warm_start import (
    TRAINING_LATEST_NAME,
    WARM_START_MANIFEST_NAME,
    write_warm_start_manifest,
)
from src.utils.mlflow_utils import log_metrics


@dataclass
class TrainingLoopResult:
    best_val_loss: float
    best_val_mae: float
    best_val_iou: float
    metrics_history: Dict[str, List]
    baseline_mae: Optional[float]
    best_tile_info_so_far: Optional[dict]
    best_iou_tile_info_so_far: Optional[dict]
    best_tile_loss_so_far: float
    best_iou_so_far: float
    best_iou_tile_loss_so_far: Optional[float]
    last_epoch: int


@dataclass
class TrainingLoopConfig:
    targets_dir: Path
    layer_registry: LayerRegistry
    tile_size: int
    target_mode: str
    binary_threshold: float
    iou_threshold: float
    early_stopping_patience: Optional[int]
    early_stopping_min_delta: float
    max_grad_norm: Optional[float]
    best_model_path: Path
    loss_plot_path: Optional[Path]
    run_name: str
    train_subsample_ratio: float
    models_dir: Optional[Path] = None
    project_root: Optional[Path] = None
    config_path: Optional[Path] = None
    mode: str = "production"
    config_snapshot: Optional[dict] = None
    representative_tiles: List[dict] = field(default_factory=list)
    viz_interval_seconds: float = 3600.0


@dataclass
class _EpochTracker:
    best_val_loss: float = float("inf")
    best_val_mae: float = float("inf")
    best_val_iou: float = 0.0
    best_val_loss_for_early_stop: float = float("inf")
    early_stopping_counter: int = 0
    baseline_mae: Optional[float] = None
    encoder_unfrozen: bool = False
    best_tile_loss_so_far: float = float("inf")
    best_tile_info_so_far: Optional[dict] = None
    best_iou_so_far: float = -1.0
    best_iou_tile_info_so_far: Optional[dict] = None
    best_iou_tile_loss_so_far: Optional[float] = None
    last_time_saved: Optional[float] = None
    last_drawn_best_tile_id: Optional[str] = None
    last_drawn_best_tile_loss: Optional[float] = None
    last_drawn_best_iou_tile_id: Optional[str] = None
    last_drawn_best_iou: Optional[float] = None
    last_representative_tiles_time: Optional[float] = None
    metrics_history: Dict[str, List] = field(default_factory=lambda: {
        "epochs": [], "train_loss": [], "val_loss": [],
        "val_mae": [], "val_iou": [], "learning_rate": [],
        "improvement_percent": [], "early_stop_counter": [],
    })

    @classmethod
    def from_resume_state(cls, state: Dict[str, Any]) -> "_EpochTracker":
        tracker = cls()
        tracker.metrics_history = copy.deepcopy(state["metrics_history"])
        tracker.early_stopping_counter = int(state["early_stopping_counter"])
        tracker.best_val_loss_for_early_stop = float(state["best_val_loss_for_early_stop"])
        tracker.best_val_loss = float(state["best_val_loss"])
        tracker.best_val_mae = float(state["best_val_mae"])
        tracker.best_val_iou = float(state["best_val_iou"])
        bl = state.get("baseline_mae")
        tracker.baseline_mae = None if bl is None else float(bl)
        tracker.encoder_unfrozen = bool(state["encoder_unfrozen"])
        return tracker


SAVE_THROTTLE_SECONDS = 3 * 60


def _build_epoch_train_loader(
    train_tiles: List[dict],
    val_tiles: List[dict],
    train_subsample_ratio: float,
    epoch: int,
    create_dataloaders_fn: Callable[[List, List], Tuple[DataLoader, DataLoader]],
) -> DataLoader:
    rng = np.random.default_rng(epoch)
    n = max(1, int(len(train_tiles) * train_subsample_ratio))
    indices = rng.choice(len(train_tiles), size=n, replace=False)
    epoch_tiles = [train_tiles[i] for i in indices]
    train_loader, _ = create_dataloaders_fn(epoch_tiles, val_tiles)
    return train_loader


def _build_training_loop_state(tracker: _EpochTracker, epoch: int) -> Dict[str, Any]:
    return {
        "metrics_history": {k: list(v) for k, v in tracker.metrics_history.items()},
        "early_stopping_counter": int(tracker.early_stopping_counter),
        "best_val_loss_for_early_stop": float(tracker.best_val_loss_for_early_stop),
        "best_val_loss": float(tracker.best_val_loss),
        "best_val_mae": float(tracker.best_val_mae),
        "best_val_iou": float(tracker.best_val_iou),
        "baseline_mae": None if tracker.baseline_mae is None else float(tracker.baseline_mae),
        "encoder_unfrozen": bool(tracker.encoder_unfrozen),
        "last_completed_epoch": int(epoch),
    }


def _handle_optuna_pruning(
    trial: Any, optuna_module: Any,
    val_metrics: dict, epoch: int, run_name: str,
) -> None:
    import mlflow
    if optuna_module is None:
        raise ImportError("Optuna is required for hyperparameter tuning. Install with: poetry install")
    trial.report(val_metrics["val_loss"], epoch)
    if not trial.should_prune():
        return
    logger.info("Trial %s pruned at epoch %s", trial.number, epoch)
    if hasattr(trial, "set_user_attr"):
        try:
            trial.set_user_attr("pruned_epoch", int(epoch))
            trial.set_user_attr("pruned_val_loss", float(val_metrics["val_loss"]))
        except Exception:
            logger.debug("Failed to set pruned trial attributes", exc_info=True)
    active = mlflow.active_run()
    if active is not None:
        logger.info(
            "OPTUNA PRUNED trial=%s epoch=%s experiment_id=%s run_id=%s run_name=%s val_loss=%.6f",
            trial.number, epoch, active.info.experiment_id,
            active.info.run_id, run_name, val_metrics["val_loss"],
        )
        logger.info("tracking_uri=%s", mlflow.get_tracking_uri())
    raise optuna_module.TrialPruned()


def _update_early_stopping(
    tracker: _EpochTracker, val_loss: float,
    patience: Optional[int], min_delta: float, epoch: int,
) -> bool:
    if not patience:
        tracker.metrics_history["early_stop_counter"].append(0)
        return False
    if val_loss < tracker.best_val_loss_for_early_stop - min_delta:
        tracker.best_val_loss_for_early_stop = val_loss
        tracker.early_stopping_counter = 0
    else:
        tracker.early_stopping_counter += 1
    tracker.metrics_history["early_stop_counter"].append(tracker.early_stopping_counter)
    if tracker.early_stopping_counter >= patience:
        logger.info(
            "EARLY STOP: No improvement for %s epochs. epoch=%s val_loss=%.6f best_val_loss=%.6f",
            patience, epoch, val_loss, tracker.best_val_loss_for_early_stop,
        )
        return True
    return False


def _check_overfit_gap(
    train_loss: float, val_loss: float,
    max_gap_ratio: Optional[float], epoch: int,
    min_epoch: int = 5,
) -> bool:
    if max_gap_ratio is None or max_gap_ratio <= 0:
        return False
    if epoch < min_epoch or train_loss <= 0:
        return False
    gap_ratio = (val_loss - train_loss) / train_loss
    if gap_ratio > max_gap_ratio:
        logger.info(
            "OVERFIT STOP: train-val gap %.0f%% exceeds %.0f%% threshold. "
            "epoch=%s train_loss=%.6f val_loss=%.6f",
            gap_ratio * 100, max_gap_ratio * 100, epoch, train_loss, val_loss,
        )
        return True
    return False


def _track_best_tiles(tracker: _EpochTracker, val_result: ValidationResult) -> None:
    if val_result.best_tile is not None:
        tile_info, tile_loss = val_result.best_tile
        if tile_loss < tracker.best_tile_loss_so_far:
            tracker.best_tile_loss_so_far = tile_loss
            tracker.best_tile_info_so_far = tile_info
            logger.info("New lowest-loss tile: %s loss=%.6f", tile_info.get("tile_id", "?"), tile_loss)
    if val_result.best_iou_tile is not None:
        if len(val_result.best_iou_tile) == 3:
            tile_info, tile_iou, tile_loss = val_result.best_iou_tile
        else:
            tile_info, tile_iou = val_result.best_iou_tile
            tile_loss = None
        if tile_iou > tracker.best_iou_so_far:
            tracker.best_iou_so_far = tile_iou
            tracker.best_iou_tile_info_so_far = tile_info
            tracker.best_iou_tile_loss_so_far = tile_loss
            logger.info("New highest IoU tile: %s IoU=%.4f", tile_info.get("tile_id", "?"), tile_iou)


def _save_best_model_and_viz(
    tracker: _EpochTracker,
    model: nn.Module, optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any],
    val_metrics: dict, epoch: int,
    cfg: TrainingLoopConfig, trial: Optional[Any],
) -> None:
    import mlflow
    import matplotlib.pyplot as plt

    prev_best = tracker.best_val_loss
    if val_metrics["val_loss"] >= prev_best:
        return
    tracker.best_val_loss = val_metrics["val_loss"]
    tracker.best_val_mae = val_metrics["val_mae"]
    tracker.best_val_iou = val_metrics.get("val_iou", 0.0)
    now = time.time()
    may_save = tracker.last_time_saved is None or (now - tracker.last_time_saved) >= SAVE_THROTTLE_SECONDS
    if may_save:
        loop_state = _build_training_loop_state(tracker, epoch)
        save_training_checkpoint(
            cfg.best_model_path, model, optimizer, epoch, val_metrics,
            lr_scheduler=lr_scheduler, training_loop_state=loop_state,
        )
        logger.info(
            "New best model saved! val_loss: %.4f | val_mae: %.4f | val_iou: %.4f",
            tracker.best_val_loss, tracker.best_val_mae, tracker.best_val_iou,
        )
        if trial is None:
            _log_best_tile_figures(tracker, model, cfg, plt, mlflow)
        tracker.last_time_saved = time.time()
    logger.info(
        "BEST epoch=%s val_loss=%.6f (prev_best=%.6f) val_mae=%.6f val_iou=%.6f%s",
        epoch, tracker.best_val_loss, prev_best, tracker.best_val_mae, tracker.best_val_iou,
        " (saved)" if may_save else " (not saved, throttle)",
    )


def _log_best_tile_figures(tracker: _EpochTracker, model: nn.Module, cfg: TrainingLoopConfig, plt, mlflow) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tracker.best_tile_info_so_far is not None:
        bid = tracker.best_tile_info_so_far.get("tile_id")
        if bid != tracker.last_drawn_best_tile_id or tracker.best_tile_loss_so_far != tracker.last_drawn_best_tile_loss:
            fig = show_best_predicted_tile(
                model, tracker.best_tile_info_so_far,
                cfg.targets_dir, cfg.layer_registry,
                device, cfg.tile_size, cfg.iou_threshold, tracker.best_tile_loss_so_far,
                target_mode=cfg.target_mode, binary_threshold=cfg.binary_threshold,
            )
            mlflow.log_figure(fig, "plots/best_predicted_tile.png")
            if cfg.loss_plot_path is not None:
                fig.savefig(cfg.loss_plot_path.parent / "best_predicted_tile.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            tracker.last_drawn_best_tile_id = bid
            tracker.last_drawn_best_tile_loss = tracker.best_tile_loss_so_far
    if tracker.best_iou_tile_info_so_far is not None:
        iid = tracker.best_iou_tile_info_so_far.get("tile_id")
        if iid != tracker.last_drawn_best_iou_tile_id or tracker.best_iou_so_far != tracker.last_drawn_best_iou:
            fig = show_highest_iou_tile(
                model, tracker.best_iou_tile_info_so_far,
                cfg.targets_dir, cfg.layer_registry,
                device, cfg.tile_size, cfg.iou_threshold, tracker.best_iou_so_far,
                target_mode=cfg.target_mode, binary_threshold=cfg.binary_threshold,
                tile_loss=tracker.best_iou_tile_loss_so_far,
            )
            mlflow.log_figure(fig, "plots/best_iou_tile.png")
            if cfg.loss_plot_path is not None:
                fig.savefig(cfg.loss_plot_path.parent / "best_iou_tile.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            tracker.last_drawn_best_iou_tile_id = iid
            tracker.last_drawn_best_iou = tracker.best_iou_so_far


def _maybe_log_representative_tiles(
    tracker: _EpochTracker, model: nn.Module, cfg: TrainingLoopConfig,
    epoch: int,
) -> None:
    if not cfg.representative_tiles:
        return
    now = time.time()
    if tracker.last_representative_tiles_time is not None:
        elapsed = now - tracker.last_representative_tiles_time
        if elapsed < cfg.viz_interval_seconds:
            return
    import mlflow
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Generating representative tile figures (epoch %s)...", epoch)
    pred_figures = create_prediction_tile_figures(
        model, cfg.representative_tiles,
        cfg.targets_dir, cfg.layer_registry, device,
        iou_threshold=cfg.iou_threshold, tile_size=cfg.tile_size,
        target_mode=cfg.target_mode, binary_threshold=cfg.binary_threshold,
    )
    for tid, fig in pred_figures.items():
        mlflow.log_figure(fig, f"prediction_tiles/{tid}.png")
        if cfg.loss_plot_path is not None:
            fig.savefig(cfg.loss_plot_path.parent / f"prediction_{tid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    channel_figures = create_representative_tiles_channel_figures(
        model, cfg.representative_tiles,
        cfg.targets_dir, cfg.layer_registry, device,
        iou_threshold=cfg.iou_threshold, tile_size=cfg.tile_size,
        target_mode=cfg.target_mode, binary_threshold=cfg.binary_threshold,
    )
    for tid, fig in channel_figures.items():
        mlflow.log_figure(fig, f"representative_channels/{tid}.png")
        if cfg.loss_plot_path is not None:
            fig.savefig(cfg.loss_plot_path.parent / f"channels_{tid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    tracker.last_representative_tiles_time = time.time()
    logger.info("Representative tiles updated (epoch %s, %d tiles)", epoch, len(cfg.representative_tiles))


def _log_epoch_summary(
    tracker: _EpochTracker, train_metrics: dict, val_metrics: dict,
    epoch: int, num_epochs: int, patience: Optional[int],
) -> None:
    summary = (
        "EPOCH %s/%s train_loss=%.6f val_loss=%.6f val_mae=%.6f val_iou=%.6f best_val_loss=%.6f"
    )
    args: list = [
        epoch, num_epochs, train_metrics["train_loss"],
        val_metrics["val_loss"], val_metrics["val_mae"],
        val_metrics.get("val_iou", 0.0), tracker.best_val_loss,
    ]
    if patience:
        summary += " early_stop=%s/%s"
        args.extend([tracker.early_stopping_counter, patience])
    if train_metrics["train_loss"] > 0:
        gap_pct = (val_metrics["val_loss"] - train_metrics["train_loss"]) / train_metrics["train_loss"] * 100
        summary += " overfit_gap=%.0f%%"
        args.append(gap_pct)
    if "val_baseline_mae" in val_metrics:
        baseline = float(val_metrics["val_baseline_mae"])
        improvement = float(val_metrics["val_improvement_over_baseline"])
        pct = (improvement / baseline * 100.0) if baseline > 0 else 0.0
        summary += " baseline_mae=%.6f improvement=%+.6f (%+.1f%%)"
        args.extend([baseline, improvement, pct])
    logger.info(summary, *args)


def _save_latest_checkpoint(
    tracker: _EpochTracker, model: nn.Module, optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any], val_metrics: dict, epoch: int,
    cfg: TrainingLoopConfig, num_epochs: int,
) -> None:
    if cfg.models_dir is None or cfg.project_root is None:
        return
    loop_state = _build_training_loop_state(tracker, epoch)
    latest_path = cfg.models_dir / TRAINING_LATEST_NAME
    save_training_checkpoint(
        latest_path, model, optimizer, epoch, val_metrics,
        lr_scheduler=lr_scheduler, training_loop_state=loop_state,
    )
    write_warm_start_manifest(
        cfg.models_dir / WARM_START_MANIFEST_NAME,
        checkpoint_path=latest_path,
        config_path=cfg.config_path,
        mode=cfg.mode,
        num_epochs_target=num_epochs,
        last_completed_epoch=epoch,
        metrics_history=tracker.metrics_history,
        best_val_loss=tracker.best_val_loss,
        best_val_mae=tracker.best_val_mae,
        best_val_iou=tracker.best_val_iou,
        loss_plot_path=cfg.loss_plot_path,
        project_root=cfg.project_root,
        config_snapshot=cfg.config_snapshot,
    )


def run_training_loop(
    config: dict,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any],
    device: torch.device,
    train_tiles: List[dict],
    val_tiles: List[dict],
    train_loader: DataLoader,
    val_loader: DataLoader,
    create_dataloaders_fn: Callable[[List, List], Tuple[DataLoader, DataLoader]],
    targets_dir: Path,
    layer_registry: LayerRegistry,
    tile_size: int,
    target_mode: str,
    binary_threshold: float,
    iou_threshold: float,
    early_stopping_patience: Optional[int],
    early_stopping_min_delta: float,
    max_grad_norm: Optional[float],
    best_model_path: Path,
    loss_plot_path: Optional[Path],
    trial: Optional[Any],
    run_name: str,
    train_subsample_ratio: float,
    optuna_module: Optional[Any],
    *,
    resume_state: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
    mode: str = "production",
    config_snapshot: Optional[dict] = None,
    representative_tiles: Optional[List[dict]] = None,
    viz_interval_seconds: float = 3600.0,
) -> TrainingLoopResult:
    import mlflow
    import matplotlib.pyplot as plt

    cfg = TrainingLoopConfig(
        targets_dir=targets_dir, layer_registry=layer_registry,
        tile_size=tile_size,
        target_mode=target_mode, binary_threshold=binary_threshold,
        iou_threshold=iou_threshold, early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta, max_grad_norm=max_grad_norm,
        best_model_path=best_model_path, loss_plot_path=loss_plot_path,
        run_name=run_name, train_subsample_ratio=train_subsample_ratio,
        models_dir=models_dir, project_root=project_root,
        config_path=config_path, mode=mode, config_snapshot=config_snapshot,
        representative_tiles=representative_tiles or [],
        viz_interval_seconds=viz_interval_seconds,
    )
    num_epochs = config["training"]["num_epochs"]
    unfreeze_after_epoch = config["model"].get("encoder", {}).get("unfreeze_after_epoch", 0)
    max_overfit_gap_ratio = config["training"].get("max_overfit_gap_ratio")

    if resume_state:
        tracker = _EpochTracker.from_resume_state(resume_state)
        start_epoch = int(resume_state["last_completed_epoch"]) + 1
        logger.info("Resuming training from epoch %s", start_epoch)
    else:
        tracker = _EpochTracker()
        start_epoch = 1

    if start_epoch > num_epochs:
        logger.warning("Nothing to train: next epoch would be %s but num_epochs is %s.", start_epoch, num_epochs)
        return _build_result(tracker, resume_state)

    current_train_loader = train_loader
    epoch = start_epoch

    for epoch in range(start_epoch, num_epochs + 1):
        if cfg.train_subsample_ratio < 1.0:
            current_train_loader = _build_epoch_train_loader(
                train_tiles, val_tiles, cfg.train_subsample_ratio, epoch, create_dataloaders_fn,
            )
        _maybe_unfreeze_encoder(model, optimizer, tracker, epoch, unfreeze_after_epoch)

        train_metrics = train_one_epoch(
            model, current_train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=cfg.max_grad_norm,
        )
        val_result = validate(
            model, val_loader, criterion, device, epoch,
            iou_threshold=cfg.iou_threshold,
            val_tile_list=val_tiles,
            return_best_tile=(trial is None),
            return_batch_losses=False,
        )
        val_metrics = val_result.metrics

        _record_metrics(tracker, epoch, train_metrics, val_metrics)

        if trial is not None:
            _handle_optuna_pruning(trial, optuna_module, val_metrics, epoch, cfg.run_name)

        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["val_loss"])
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
        tracker.metrics_history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        should_stop = _update_early_stopping(
            tracker, val_metrics["val_loss"], cfg.early_stopping_patience, cfg.early_stopping_min_delta, epoch,
        )
        if not should_stop:
            should_stop = _check_overfit_gap(
                train_metrics["train_loss"], val_metrics["val_loss"],
                max_overfit_gap_ratio, epoch,
            )

        log_metrics({**train_metrics, **val_metrics}, step=epoch)

        if trial is None:
            _track_best_tiles(tracker, val_result)

        _save_best_model_and_viz(tracker, model, optimizer, lr_scheduler, val_metrics, epoch, cfg, trial)
        _log_epoch_summary(tracker, train_metrics, val_metrics, epoch, num_epochs, cfg.early_stopping_patience)
        _log_loss_plot(tracker, cfg, mlflow, plt)
        if trial is None:
            _maybe_log_representative_tiles(tracker, model, cfg, epoch)
        _save_latest_checkpoint(tracker, model, optimizer, lr_scheduler, val_metrics, epoch, cfg, num_epochs)

        if should_stop:
            break

    return TrainingLoopResult(
        best_val_loss=tracker.best_val_loss,
        best_val_mae=tracker.best_val_mae,
        best_val_iou=tracker.best_val_iou,
        metrics_history=tracker.metrics_history,
        baseline_mae=tracker.baseline_mae,
        best_tile_info_so_far=tracker.best_tile_info_so_far,
        best_iou_tile_info_so_far=tracker.best_iou_tile_info_so_far,
        best_tile_loss_so_far=tracker.best_tile_loss_so_far,
        best_iou_so_far=tracker.best_iou_so_far,
        best_iou_tile_loss_so_far=tracker.best_iou_tile_loss_so_far,
        last_epoch=epoch,
    )


def _maybe_unfreeze_encoder(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    tracker: _EpochTracker, epoch: int, unfreeze_after_epoch: int,
) -> None:
    if (unfreeze_after_epoch > 0 and epoch == unfreeze_after_epoch
            and hasattr(model, "unfreeze_encoder") and not tracker.encoder_unfrozen):
        logger.info("Unfreezing encoder at epoch %s", epoch)
        model.unfreeze_encoder()
        tracker.encoder_unfrozen = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1
        logger.info("Reduced learning rate to %s for fine-tuning", optimizer.param_groups[0]["lr"])


def _record_metrics(tracker: _EpochTracker, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
    tracker.metrics_history["epochs"].append(epoch)
    tracker.metrics_history["train_loss"].append(train_metrics["train_loss"])
    tracker.metrics_history["val_loss"].append(val_metrics["val_loss"])
    tracker.metrics_history["val_mae"].append(val_metrics["val_mae"])
    tracker.metrics_history["val_iou"].append(val_metrics["val_iou"])
    if "val_baseline_mae" in val_metrics:
        if tracker.baseline_mae is None:
            tracker.baseline_mae = val_metrics["val_baseline_mae"]
        improvement = val_metrics["val_improvement_over_baseline"]
        improvement_percent = (improvement / tracker.baseline_mae) * 100 if tracker.baseline_mae > 0 else 0.0
        tracker.metrics_history["improvement_percent"].append(improvement_percent)


def _log_loss_plot(tracker: _EpochTracker, cfg: TrainingLoopConfig, mlflow, plt) -> None:
    loss_fig = plot_loss_simple(
        tracker.metrics_history["epochs"],
        tracker.metrics_history["train_loss"],
        tracker.metrics_history["val_loss"],
        learning_rate=tracker.metrics_history["learning_rate"],
    )
    mlflow.log_figure(loss_fig, "plots/loss.png")
    if cfg.loss_plot_path is not None:
        loss_fig.savefig(cfg.loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(loss_fig)


def _build_result(tracker: _EpochTracker, resume_state: Optional[Dict[str, Any]]) -> TrainingLoopResult:
    tls = resume_state or {}
    le = int(tls.get("last_completed_epoch", 0))
    return TrainingLoopResult(
        best_val_loss=float(tls.get("best_val_loss", float("inf"))),
        best_val_mae=float(tls.get("best_val_mae", float("inf"))),
        best_val_iou=float(tls.get("best_val_iou", 0.0)),
        metrics_history=tracker.metrics_history,
        baseline_mae=tracker.baseline_mae,
        best_tile_info_so_far=tracker.best_tile_info_so_far,
        best_iou_tile_info_so_far=tracker.best_iou_tile_info_so_far,
        best_tile_loss_so_far=tracker.best_tile_loss_so_far,
        best_iou_so_far=tracker.best_iou_so_far,
        best_iou_tile_loss_so_far=tracker.best_iou_tile_loss_so_far,
        last_epoch=le,
    )
