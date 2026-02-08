"""
Visualization utilities for training metrics.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch


def _calculate_smart_ylim(
    values: List[float],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    padding: float = 0.1,
) -> Tuple[float, float]:
    """
    Calculate Y-axis limits excluding outliers.

    Args:
        values: List of values
        lower_percentile: Lower percentile to use (default: 5th)
        upper_percentile: Upper percentile to use (default: 95th)
        padding: Padding factor (default: 10% on each side)

    Returns:
        Tuple of (ymin, ymax)
    """
    if not values:
        return 0.0, 1.0

    values_array = np.array(values)

    # Calculate percentiles
    y_min = np.percentile(values_array, lower_percentile)
    y_max = np.percentile(values_array, upper_percentile)

    # Add padding
    y_range = y_max - y_min
    if y_range > 0:
        y_min = max(0, y_min - padding * y_range)  # Don't go below 0 for most metrics
        y_max = y_max + padding * y_range
    else:
        # If all values are the same, add small padding
        y_min = max(0, y_min - abs(y_min) * padding)
        y_max = y_max + abs(y_max) * padding

    return float(y_min), float(y_max)


def plot_mae_comparison(
    epochs: List[int],
    model_mae: List[float],
    baseline_mae: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Model MAE vs Baseline MAE across epochs.

    Args:
        epochs: List of epoch numbers
        model_mae: List of model MAE values per epoch
        baseline_mae: Baseline MAE value (constant line)
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot model MAE
    ax.plot(epochs, model_mae, 'b-', linewidth=2, label='Model MAE', marker='o', markersize=4)

    # Plot baseline MAE if provided
    if baseline_mae is not None:
        ax.axhline(y=baseline_mae, color='r', linestyle='--', linewidth=2, label=f'Baseline MAE ({baseline_mae:.4f})')

    # Set Y-axis limits excluding outliers
    y_min, y_max = _calculate_smart_ylim(model_mae)
    if baseline_mae is not None:
        # Include baseline in range calculation
        all_values = model_mae + [baseline_mae]
        y_min, y_max = _calculate_smart_ylim(all_values)
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Model MAE vs Baseline MAE', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(epochs), right=max(epochs))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_loss(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation loss across epochs.

    Args:
        epochs: List of epoch numbers
        train_loss: List of training loss values per epoch
        val_loss: List of validation loss values per epoch
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)

    # Set Y-axis limits excluding outliers (combine both loss series)
    all_loss_values = train_loss + val_loss
    y_min, y_max = _calculate_smart_ylim(all_loss_values)
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(epochs), right=max(epochs))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_iou(
    epochs: List[int],
    val_iou: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot validation IoU across epochs.

    Args:
        epochs: List of epoch numbers
        val_iou: List of validation IoU values per epoch
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, val_iou, 'g-', linewidth=2, label='Val IoU', marker='o', markersize=4)

    # Set Y-axis limits excluding outliers (IoU is 0-1, but use smart limits)
    y_min, y_max = _calculate_smart_ylim(val_iou)
    # Ensure IoU stays in [0, 1] range
    y_min = max(0, y_min)
    y_max = min(1.0, max(1.0, y_max))  # At least show up to 1.0 if values are close

    ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Validation IoU', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(epochs), right=max(epochs))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_improvement_percentage(
    epochs: List[int],
    improvement_percent: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot percent improvement over baseline across epochs.

    Args:
        epochs: List of epoch numbers
        improvement_percent: List of percent improvement values (positive = better than baseline)
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['g' if p >= 0 else 'r' for p in improvement_percent]
    ax.bar(epochs, improvement_percent, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Set Y-axis limits excluding outliers
    y_min, y_max = _calculate_smart_ylim(improvement_percent, padding=0.15)
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Percent Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(left=min(epochs) - 0.5, right=max(epochs) + 0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_training_plots(
    metrics_history: Dict[str, List[float]],
    baseline_mae: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Create all training plots from metrics history.

    Args:
        metrics_history: Dictionary with keys like 'epochs', 'train_loss', 'val_loss',
                        'val_mae', 'val_iou', 'improvement_percent'
        baseline_mae: Optional baseline MAE value
        output_dir: Optional directory to save figures

    Returns:
        Dictionary mapping plot names to matplotlib Figures
    """
    epochs = metrics_history.get('epochs', [])
    if not epochs:
        return {}

    figures = {}

    # MAE comparison
    if 'val_mae' in metrics_history:
        fig = plot_mae_comparison(
            epochs, metrics_history['val_mae'], baseline_mae
        )
        figures['mae_comparison'] = fig
        if output_dir:
            fig.savefig(output_dir / 'mae_comparison.png', dpi=150, bbox_inches='tight')

    # Loss plot
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        fig = plot_loss(
            epochs, metrics_history['train_loss'], metrics_history['val_loss']
        )
        figures['loss'] = fig
        if output_dir:
            fig.savefig(output_dir / 'loss.png', dpi=150, bbox_inches='tight')

    # IoU plot
    if 'val_iou' in metrics_history:
        fig = plot_iou(epochs, metrics_history['val_iou'])
        figures['iou'] = fig
        if output_dir:
            fig.savefig(output_dir / 'iou.png', dpi=150, bbox_inches='tight')

    # Improvement percentage
    if 'improvement_percent' in metrics_history:
        fig = plot_improvement_percentage(epochs, metrics_history['improvement_percent'])
        figures['improvement_percent'] = fig
        if output_dir:
            fig.savefig(output_dir / 'improvement_percent.png', dpi=150, bbox_inches='tight')

    return figures


def get_representative_tile_ids_for_viz(
    viz_config: dict,
    mode: str,
    tile_size: int = 256,
) -> List[Union[int, str]]:
    """Return representative_tile_ids for visualization based on mode and tile size."""
    if tile_size == 512 and mode == "dev" and "representative_tile_ids_dev_512" in viz_config:
        return viz_config.get("representative_tile_ids_dev_512", [])
    if tile_size == 512 and mode != "dev" and "representative_tile_ids_512" in viz_config:
        return viz_config.get("representative_tile_ids_512", [])
    if mode == "dev" and "representative_tile_ids_dev" in viz_config:
        return viz_config.get("representative_tile_ids_dev", [])
    return viz_config.get("representative_tile_ids", [])


def _tile_id_to_index(tile_id: str) -> Optional[int]:
    parts = tile_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def resolve_representative_tiles(
    all_tiles: List[dict],
    config_ids: List[Union[int, str]],
) -> List[dict]:
    requested_indices = {x for x in config_ids if isinstance(x, int)}
    requested_ids_str = {str(x).strip() for x in config_ids if isinstance(x, str)}
    out = []
    for t in all_tiles:
        tid = t.get("tile_id")
        if tid is None:
            continue
        idx = _tile_id_to_index(tid)
        if idx is not None and idx in requested_indices:
            out.append(t)
        elif tid in requested_ids_str:
            out.append(t)
    return out


def _load_rgb_for_display(features_path: Path, features_base_dir: Path) -> np.ndarray:
    p = Path(features_path)
    path = (features_base_dir / p) if not p.is_absolute() else p
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb


def create_prediction_tile_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    iou_threshold: float = 5.0,
    tile_size: int = 256,
) -> Dict[str, plt.Figure]:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

    if rep_tiles:
        first_path = Path(features_dir) / rep_tiles[0].get("features_path", "").replace("\\", "/")
        if first_path.exists():
            with rasterio.open(first_path) as src:
                h, w = src.height, src.width
                if h == w:
                    tile_size = int(h)
    model.eval()
    dataset = TileDataset(
        rep_tiles,
        features_dir,
        targets_dir,
        normalization_stats,
        tile_size=tile_size,
    )
    figures = {}
    for i, tile_info in enumerate(rep_tiles):
        features, target = dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(features_batch)
        pred_cpu = pred.squeeze(0).cpu()
        target_cpu = target.unsqueeze(0)
        pred_np = np.squeeze(pred_cpu.numpy())
        target_np = target.squeeze().numpy()
        mae = compute_mae(pred_cpu, target_cpu)
        rmse = compute_rmse(pred_cpu, target_cpu)
        iou = compute_iou(pred_cpu, target_cpu, threshold=iou_threshold)
        metrics = {"mae": mae, "rmse": rmse, "iou": iou}
        fp = tile_info.get("features_path", "")
        rgb = _load_rgb_for_display(Path(fp.replace("\\", "/")), Path(features_dir))
        tid = tile_info.get("tile_id", f"tile_{i}")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        vmin, vmax = 0.0, 20.0
        axes[1].imshow(target_np, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1].set_title("Proximity (target)")
        axes[1].axis("off")
        axes[2].imshow(pred_np, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        title = f"Tile: {tid}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  IoU: {iou:.4f}"
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        figures[tid] = fig
    return figures
