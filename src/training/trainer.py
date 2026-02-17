"""
Training utilities.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou


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
) -> Tuple[Dict[str, float], Optional[Tuple[dict, float]], Optional[Tuple[dict, float]], Optional[List[float]]]:
    """
    Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        iou_threshold: Threshold for IoU calculation
        val_tile_list: Optional list of validation tile info (for baseline comparison)
        return_best_tile: If True and val_tile_list provided, return lowest-loss and highest-IoU tiles
        return_batch_losses: If True, return list of per-batch validation losses (for spike debugging)

    Returns:
        (metrics_dict, best_tile_result, best_iou_tile_result, batch_losses).
        batch_losses is None unless return_batch_losses=True.
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_iou = 0.0
    num_batches = 0
    batch_losses: List[float] = [] if return_batch_losses else []

    best_tile_result: Optional[Tuple[dict, float]] = None
    best_iou_tile_result: Optional[Tuple[dict, float]] = None
    tile_index = 0

    # For baseline comparison - compute aggregate baseline MAE from tile list
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

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        # Track prediction statistics for diagnostics
        all_pred_values = []
        all_target_values = []

        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(features)

            # Compute loss
            loss = criterion(outputs, targets)

            # Metrics
            mae = compute_mae(outputs, targets)
            rmse = compute_rmse(outputs, targets)
            iou = compute_iou(outputs, targets, threshold=iou_threshold)

            # Per-sample loss and IoU for best-tile tracking
            if return_best_tile and val_tile_list is not None:
                batch_size = outputs.size(0)
                for i in range(batch_size):
                    if tile_index >= len(val_tile_list):
                        break
                    sample_loss = criterion(
                        outputs[i : i + 1], targets[i : i + 1]
                    ).item()
                    sample_iou = compute_iou(
                        outputs[i : i + 1], targets[i : i + 1], threshold=iou_threshold
                    )
                    if best_tile_result is None or sample_loss < best_tile_result[1]:
                        best_tile_result = (val_tile_list[tile_index].copy(), sample_loss)
                    if best_iou_tile_result is None or sample_iou > best_iou_tile_result[1]:
                        best_iou_tile_result = (val_tile_list[tile_index].copy(), sample_iou)
                    tile_index += 1

            # Collect statistics for diagnostics
            all_pred_values.append(outputs.cpu().numpy().flatten())
            all_target_values.append(targets.cpu().numpy().flatten())

            total_loss += loss.item()
            total_mae += mae
            total_rmse += rmse
            total_iou += iou
            num_batches += 1
            if return_batch_losses:
                batch_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mae": f"{mae:.4f}",
                "iou": f"{iou:.4f}",
            })

        # Compute prediction statistics
        all_pred_values = np.concatenate(all_pred_values)
        all_target_values = np.concatenate(all_target_values)

        pred_min = float(np.min(all_pred_values))
        pred_max = float(np.max(all_pred_values))
        pred_mean = float(np.mean(all_pred_values))
        pred_median = float(np.median(all_pred_values))

        target_min = float(np.min(all_target_values))
        target_max = float(np.max(all_target_values))
        target_mean = float(np.mean(all_target_values))

        # Count predictions above threshold
        pred_above_threshold = np.sum(all_pred_values >= iou_threshold)
        target_above_threshold = np.sum(all_target_values >= iou_threshold)

        print(f"\n  [DIAGNOSTICS] Prediction stats: min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}, median={pred_median:.3f}")
        print(f"  [DIAGNOSTICS] Target stats: min={target_min:.3f}, max={target_max:.3f}, mean={target_mean:.3f}")
        print(f"  [DIAGNOSTICS] Pixels >= {iou_threshold}: pred={pred_above_threshold:,} ({pred_above_threshold/len(all_pred_values)*100:.2f}%), target={target_above_threshold:,} ({target_above_threshold/len(all_target_values)*100:.2f}%)")

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_iou = total_iou / num_batches

    metrics = {
        "val_loss": avg_loss,
        "val_mae": avg_mae,
        "val_rmse": avg_rmse,
        "val_iou": avg_iou,
        "val_pred_min": pred_min,
        "val_pred_max": pred_max,
        "val_pred_mean": pred_mean,
    }

    # Add baseline comparison if tile info available
    if baseline_tile_count > 0:
        avg_baseline_mae = baseline_mae_sum / baseline_tile_count
        improvement = avg_baseline_mae - avg_mae
        is_better = avg_mae < avg_baseline_mae

        metrics["val_baseline_mae"] = avg_baseline_mae
        metrics["val_improvement_over_baseline"] = improvement
        metrics["val_better_than_baseline"] = float(is_better)  # 1.0 if better, 0.0 if not

    out_batch_losses = batch_losses if return_batch_losses else None
    return (metrics, best_tile_result, best_iou_tile_result, out_batch_losses)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, checkpoint_path)
