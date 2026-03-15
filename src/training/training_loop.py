"""
Training loop: one-epoch train/validate, Optuna pruning, early stopping, best-model save, loss plot.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import train_one_epoch, validate, save_checkpoint
from src.training.visualization import plot_loss_simple, show_best_predicted_tile, show_highest_iou_tile
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
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    tile_size: int,
    target_mode: str,
    binary_threshold: float,
    segmentation_dir: Optional[Path],
    slope_stripes_channel_dir: Optional[Path],
    use_rgb: bool,
    use_dem: bool,
    use_slope: bool,
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
) -> TrainingLoopResult:
    num_epochs = config["training"]["num_epochs"]
    unfreeze_after_epoch = config["model"].get("encoder", {}).get("unfreeze_after_epoch", 0)
    encoder_unfrozen = False
    SAVE_THROTTLE_SECONDS = 3 * 60

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_val_iou = 0.0
    best_tile_loss_so_far = float("inf")
    best_tile_info_so_far = None
    best_iou_so_far = -1.0
    best_iou_tile_info_so_far = None
    best_iou_tile_loss_so_far = None
    last_time_saved = None
    last_drawn_best_tile_id = None
    last_drawn_best_tile_loss = None
    last_drawn_best_iou_tile_id = None
    last_drawn_best_iou = None
    early_stopping_counter = 0
    best_val_loss_for_early_stop = float("inf")
    baseline_mae = None

    metrics_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_iou": [],
        "improvement_percent": [],
        "early_stop_counter": [],
    }

    import logging
    import mlflow
    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)
    current_train_loader = train_loader

    for epoch in range(1, num_epochs + 1):
        if train_subsample_ratio < 1.0:
            current_train_loader = _build_epoch_train_loader(
                train_tiles, val_tiles, train_subsample_ratio, epoch, create_dataloaders_fn,
            )

        if (unfreeze_after_epoch > 0 and epoch == unfreeze_after_epoch and
                hasattr(model, "unfreeze_encoder") and not encoder_unfrozen):
            logger.info("Unfreezing encoder at epoch %s", epoch)
            model.unfreeze_encoder()
            encoder_unfrozen = True
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1
            logger.info("Reduced learning rate to %s for fine-tuning", optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model, current_train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=max_grad_norm,
        )
        val_metrics, best_tile_result, best_iou_tile_result, _ = validate(
            model, val_loader, criterion, device, epoch,
            iou_threshold=iou_threshold,
            val_tile_list=val_tiles,
            return_best_tile=(trial is None),
            return_batch_losses=False,
        )

        metrics_history["epochs"].append(epoch)
        metrics_history["train_loss"].append(train_metrics["train_loss"])
        metrics_history["val_loss"].append(val_metrics["val_loss"])
        metrics_history["val_mae"].append(val_metrics["val_mae"])
        metrics_history["val_iou"].append(val_metrics["val_iou"])
        if "val_baseline_mae" in val_metrics:
            if baseline_mae is None:
                baseline_mae = val_metrics["val_baseline_mae"]
            improvement = val_metrics["val_improvement_over_baseline"]
            improvement_percent = (improvement / baseline_mae) * 100 if baseline_mae > 0 else 0.0
            metrics_history["improvement_percent"].append(improvement_percent)

        if trial is not None:
            if optuna_module is None:
                raise ImportError("Optuna is required for hyperparameter tuning. Install with: poetry install")
            trial.report(val_metrics["val_loss"], epoch)
            if trial.should_prune():
                logger.info("Trial %s pruned at epoch %s", trial.number, epoch)
                if hasattr(trial, "set_user_attr"):
                    try:
                        trial.set_user_attr("pruned_epoch", int(epoch))
                        trial.set_user_attr("pruned_val_loss", float(val_metrics["val_loss"]))
                    except Exception:
                        pass
                active = mlflow.active_run()
                if active is not None:
                    print(
                        f"[MLFLOW][OPTUNA][PRUNED] trial={trial.number} epoch={epoch} "
                        f"experiment_id={active.info.experiment_id} run_id={active.info.run_id} run_name={run_name} "
                        f"val_loss={val_metrics['val_loss']:.6f}",
                        flush=True,
                    )
                    print(f"[MLFLOW] tracking_uri={mlflow.get_tracking_uri()}", flush=True)
                raise optuna_module.TrialPruned()

        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["val_loss"])
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

        if early_stopping_patience:
            if val_metrics["val_loss"] < best_val_loss_for_early_stop - early_stopping_min_delta:
                best_val_loss_for_early_stop = val_metrics["val_loss"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            metrics_history["early_stop_counter"].append(early_stopping_counter)
            if early_stopping_counter >= early_stopping_patience:
                logger.info("[EARLY STOP] No improvement for %s epochs. Stopping training.", early_stopping_patience)
                print(
                    f"[EARLY STOP] epoch={epoch} val_loss={val_metrics['val_loss']:.6f} "
                    f"best_val_loss={best_val_loss_for_early_stop:.6f} patience={early_stopping_patience}",
                    flush=True,
                )
                break
        else:
            metrics_history["early_stop_counter"].append(0)

        log_metrics({**train_metrics, **val_metrics}, step=epoch)

        if trial is None and best_tile_result is not None:
            tile_info, tile_loss = best_tile_result
            if tile_loss < best_tile_loss_so_far:
                best_tile_loss_so_far = tile_loss
                best_tile_info_so_far = tile_info
                logger.info("New lowest-loss tile: %s loss=%.6f", tile_info.get("tile_id", "?"), best_tile_loss_so_far)
        if trial is None and best_iou_tile_result is not None:
            if len(best_iou_tile_result) == 3:
                tile_info, tile_iou, tile_loss = best_iou_tile_result
            else:
                tile_info, tile_iou = best_iou_tile_result
                tile_loss = None
            if tile_iou > best_iou_so_far:
                best_iou_so_far = tile_iou
                best_iou_tile_info_so_far = tile_info
                best_iou_tile_loss_so_far = tile_loss
                logger.info("New highest IoU tile: %s IoU=%.4f", tile_info.get("tile_id", "?"), best_iou_so_far)

        if val_metrics["val_loss"] < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = val_metrics["val_loss"]
            best_val_mae = val_metrics["val_mae"]
            best_val_iou = val_metrics.get("val_iou", 0.0)
            now = time.time()
            may_save = last_time_saved is None or (now - last_time_saved) >= SAVE_THROTTLE_SECONDS
            if may_save:
                save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
                logger.info(
                    "New best model saved! val_loss: %.4f | val_mae: %.4f | val_iou: %.4f",
                    best_val_loss, best_val_mae, best_val_iou,
                )
                if trial is None and best_tile_info_so_far is not None:
                    bid = best_tile_info_so_far.get("tile_id")
                    if bid != last_drawn_best_tile_id or best_tile_loss_so_far != last_drawn_best_tile_loss:
                        fig = show_best_predicted_tile(
                            model, best_tile_info_so_far, features_dir, targets_dir, normalization_stats,
                            device, tile_size, iou_threshold, best_tile_loss_so_far,
                            target_mode=target_mode, binary_threshold=binary_threshold,
                            segmentation_base_dir=segmentation_dir, slope_stripes_base_dir=slope_stripes_channel_dir,
                            use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
                        )
                        mlflow.log_figure(fig, "plots/best_predicted_tile.png")
                        if loss_plot_path is not None:
                            fig.savefig(loss_plot_path.parent / "best_predicted_tile.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        last_drawn_best_tile_id = bid
                        last_drawn_best_tile_loss = best_tile_loss_so_far
                if trial is None and best_iou_tile_info_so_far is not None:
                    iid = best_iou_tile_info_so_far.get("tile_id")
                    if iid != last_drawn_best_iou_tile_id or best_iou_so_far != last_drawn_best_iou:
                        fig = show_highest_iou_tile(
                            model, best_iou_tile_info_so_far, features_dir, targets_dir, normalization_stats,
                            device, tile_size, iou_threshold, best_iou_so_far,
                            target_mode=target_mode, binary_threshold=binary_threshold,
                            segmentation_base_dir=segmentation_dir, slope_stripes_base_dir=slope_stripes_channel_dir,
                            tile_loss=best_iou_tile_loss_so_far, use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
                        )
                        mlflow.log_figure(fig, "plots/best_iou_tile.png")
                        if loss_plot_path is not None:
                            fig.savefig(loss_plot_path.parent / "best_iou_tile.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        last_drawn_best_iou_tile_id = iid
                        last_drawn_best_iou = best_iou_so_far
                last_time_saved = time.time()
            print(
                f"[BEST] epoch={epoch} val_loss={best_val_loss:.6f} (prev_best={prev_best:.6f}) "
                f"val_mae={best_val_mae:.6f} val_iou={best_val_iou:.6f}" + (" (saved)" if may_save else " (not saved, throttle)"),
                flush=True,
            )

        summary = (
            f"[EPOCH] {epoch}/{num_epochs} train_loss={train_metrics['train_loss']:.6f} "
            f"val_loss={val_metrics['val_loss']:.6f} val_mae={val_metrics['val_mae']:.6f} "
            f"val_iou={val_metrics.get('val_iou', 0.0):.6f} best_val_loss={best_val_loss:.6f}"
        )
        if early_stopping_patience:
            summary += f" early_stop={early_stopping_counter}/{early_stopping_patience}"
        if "val_baseline_mae" in val_metrics:
            baseline = float(val_metrics["val_baseline_mae"])
            improvement = float(val_metrics["val_improvement_over_baseline"])
            pct = (improvement / baseline * 100.0) if baseline > 0 else 0.0
            summary += f" baseline_mae={baseline:.6f} improvement={improvement:+.6f} ({pct:+.1f}%)"
        print(summary, flush=True)

        loss_fig = plot_loss_simple(
            metrics_history["epochs"], metrics_history["train_loss"], metrics_history["val_loss"],
        )
        mlflow.log_figure(loss_fig, "plots/loss.png")
        if loss_plot_path is not None:
            loss_fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close(loss_fig)

    return TrainingLoopResult(
        best_val_loss=best_val_loss,
        best_val_mae=best_val_mae,
        best_val_iou=best_val_iou,
        metrics_history=metrics_history,
        baseline_mae=baseline_mae,
        best_tile_info_so_far=best_tile_info_so_far,
        best_iou_tile_info_so_far=best_iou_tile_info_so_far,
        best_tile_loss_so_far=best_tile_loss_so_far,
        best_iou_so_far=best_iou_so_far,
        best_iou_tile_loss_so_far=best_iou_tile_loss_so_far,
        last_epoch=epoch,
    )
