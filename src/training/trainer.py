"""
Training utilities.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    metrics: Dict[str, float]
    best_tile: Optional[Tuple[dict, float]] = None
    best_iou_tile: Optional[Tuple[dict, float, float]] = None
    batch_losses: Optional[List[float]] = None


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        max_grad_norm: Optional maximum gradient norm for clipping

    Returns:
        Dictionary of training metrics
    """
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for features, targets in pbar:
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Metrics
        with torch.no_grad():
            mae = compute_mae(outputs, targets)
            rmse = compute_rmse(outputs, targets)

        total_loss += loss.item()
        total_mae += mae
        total_rmse += rmse
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mae": f"{mae:.4f}",
        })

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches

    return {
        "train_loss": avg_loss,
        "train_mae": avg_mae,
        "train_rmse": avg_rmse,
    }


def _compute_baseline_mae(val_tile_list: Optional[List[dict]]) -> Tuple[float, int]:
    baseline_mae_sum = 0.0
    baseline_tile_count = 0
    if val_tile_list:
        for tile_info in val_tile_list:
            baseline_metrics = tile_info.get("target_stats", {}).get("baseline_metrics", {})
            if baseline_metrics:
                baseline_mae = baseline_metrics.get("baseline_mae", {}).get("predict_zero", None)
                if baseline_mae is not None:
                    baseline_mae_sum += baseline_mae
                    baseline_tile_count += 1
    return baseline_mae_sum, baseline_tile_count


def _track_best_tiles(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    val_tile_list: List[dict],
    tile_index: int,
    iou_threshold: float,
    best_tile_result: Optional[Tuple[dict, float]],
    best_iou_tile_result: Optional[Tuple[dict, float, float]],
) -> Tuple[int, Optional[Tuple[dict, float]], Optional[Tuple[dict, float, float]]]:
    batch_size = outputs.size(0)
    for i in range(batch_size):
        if tile_index >= len(val_tile_list):
            break
        sample_loss = criterion(outputs[i : i + 1], targets[i : i + 1]).item()
        sample_iou = compute_iou(outputs[i : i + 1], targets[i : i + 1], threshold=iou_threshold)
        if best_tile_result is None or sample_loss < best_tile_result[1]:
            best_tile_result = (val_tile_list[tile_index].copy(), sample_loss)
        if best_iou_tile_result is None or sample_iou > best_iou_tile_result[1]:
            best_iou_tile_result = (val_tile_list[tile_index].copy(), sample_iou, sample_loss)
        tile_index += 1
    return tile_index, best_tile_result, best_iou_tile_result


def _log_prediction_diagnostics(
    all_pred_values: np.ndarray,
    all_target_values: np.ndarray,
    iou_threshold: float,
) -> None:
    pred_min = float(np.min(all_pred_values))
    pred_max = float(np.max(all_pred_values))
    pred_mean = float(np.mean(all_pred_values))
    pred_median = float(np.median(all_pred_values))
    target_min = float(np.min(all_target_values))
    target_max = float(np.max(all_target_values))
    target_mean = float(np.mean(all_target_values))
    pred_above = np.sum(all_pred_values >= iou_threshold)
    target_above = np.sum(all_target_values >= iou_threshold)
    n_pred = len(all_pred_values)
    n_target = len(all_target_values)
    logger.info(
        "Prediction stats: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
        pred_min, pred_max, pred_mean, pred_median,
    )
    logger.info(
        "Target stats: min=%.3f, max=%.3f, mean=%.3f",
        target_min, target_max, target_mean,
    )
    logger.info(
        "Pixels >= %.1f: pred=%d (%.2f%%), target=%d (%.2f%%)",
        iou_threshold, pred_above, pred_above / n_pred * 100,
        target_above, target_above / n_target * 100,
    )


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    iou_threshold: float = 5.0,
    val_tile_list: Optional[List[dict]] = None,
    return_best_tile: bool = False,
    return_batch_losses: bool = False,
) -> ValidationResult:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_iou = 0.0
    num_batches = 0
    batch_losses: List[float] = []

    best_tile_result: Optional[Tuple[dict, float]] = None
    best_iou_tile_result: Optional[Tuple[dict, float, float]] = None
    tile_index = 0

    baseline_mae_sum, baseline_tile_count = _compute_baseline_mae(val_tile_list)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        all_pred_values = []
        all_target_values = []

        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            mae = compute_mae(outputs, targets)
            rmse = compute_rmse(outputs, targets)
            iou = compute_iou(outputs, targets, threshold=iou_threshold)

            if return_best_tile and val_tile_list is not None:
                tile_index, best_tile_result, best_iou_tile_result = _track_best_tiles(
                    outputs, targets, criterion, val_tile_list, tile_index, iou_threshold,
                    best_tile_result, best_iou_tile_result,
                )

            all_pred_values.append(outputs.cpu().numpy().flatten())
            all_target_values.append(targets.cpu().numpy().flatten())

            total_loss += loss.item()
            total_mae += mae
            total_rmse += rmse
            total_iou += iou
            num_batches += 1
            if return_batch_losses:
                batch_losses.append(loss.item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mae": f"{mae:.4f}",
                "iou": f"{iou:.4f}",
            })

        all_preds = np.concatenate(all_pred_values)
        all_targets = np.concatenate(all_target_values)
        _log_prediction_diagnostics(all_preds, all_targets, iou_threshold)

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_iou = total_iou / num_batches

    metrics = {
        "val_loss": avg_loss,
        "val_mae": avg_mae,
        "val_rmse": avg_rmse,
        "val_iou": avg_iou,
        "val_pred_min": float(np.min(all_preds)),
        "val_pred_max": float(np.max(all_preds)),
        "val_pred_mean": float(np.mean(all_preds)),
    }

    if baseline_tile_count > 0:
        avg_baseline_mae = baseline_mae_sum / baseline_tile_count
        improvement = avg_baseline_mae - avg_mae
        metrics["val_baseline_mae"] = avg_baseline_mae
        metrics["val_improvement_over_baseline"] = improvement
        metrics["val_better_than_baseline"] = float(avg_mae < avg_baseline_mae)

    return ValidationResult(
        metrics=metrics,
        best_tile=best_tile_result,
        best_iou_tile=best_iou_tile_result,
        batch_losses=batch_losses if return_batch_losses else None,
    )


def save_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    *,
    lr_scheduler: Optional[Any] = None,
    training_loop_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint (model, optimizer, optional scheduler, optional full loop state for resume).
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if training_loop_state is not None:
        payload["training_loop_state"] = training_loop_state
    torch.save(payload, checkpoint_path)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
) -> None:
    """Backward-compatible thin wrapper (weights + optimizer only)."""
    save_training_checkpoint(
        checkpoint_path, model, optimizer, epoch, metrics,
    )


def load_training_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a saved training checkpoint (model + optional optimizer, scheduler, loop state).

    Always uses ``map_location=\"cpu\"``. Loading with ``map_location=cuda`` would place
    **all** tensors (including large Adam momentum buffers in ``optimizer_state_dict``)
    on the GPU during unpickling—even when only ``model_state_dict`` is used (e.g. init-weights).
    That wastes VRAM/PCIe time and can stall for minutes. Callers apply state to modules
    already on ``device`` via ``load_state_dict`` (PyTorch copies weights to the right device).
    """
    _ = device
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
