"""
Visualization utilities for training metrics.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Sequence

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
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


def _xlim_epochs(epochs: List[int]) -> Tuple[float, float]:
    """X limits for epoch axis; avoid singular when len(epochs)==1."""
    if not epochs:
        return 0.0, 1.0
    lo, hi = min(epochs), max(epochs)
    if lo == hi:
        return float(lo) - 0.5, float(hi) + 0.5
    return float(lo), float(hi)


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
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_loss_simple(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Minimal loss plot (train + val curves only). Use during training for fast per-epoch updates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    all_loss_values = train_loss + val_loss
    y_min, y_max = _calculate_smart_ylim(all_loss_values)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_loss(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    output_path: Optional[Path] = None,
    *,
    min_val_loss: Optional[float] = None,
    early_stop_counter: Optional[List[int]] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: Optional[float] = None,
    config_summary: Optional[str] = None,
    num_train_tiles: Optional[int] = None,
    num_val_tiles: Optional[int] = None,
    freeze_encoder: Optional[bool] = None,
    unfreeze_after_epoch: Optional[int] = None,
    training_start_datetime: Optional[str] = None,
    training_duration_seconds: Optional[float] = None,
    run_intention: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss across epochs.

    Optional: min val_loss line with label, early-stop bar (right axis),
    unfreeze vertical line (when freeze_encoder and unfreeze_after_epoch > 0),
    and small-font config/tile info.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)

    min_val = min_val_loss if min_val_loss is not None else (min(val_loss) if val_loss else None)
    if min_val is not None:
        ax.axhline(y=min_val, color='gray', linestyle=':', linewidth=1.5, label=f'min val_loss = {min_val:.4f}')
    all_loss_values = train_loss + val_loss
    if min_val is not None:
        all_loss_values = all_loss_values + [min_val]
    y_min, y_max = _calculate_smart_ylim(all_loss_values)
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    title = 'Training and Validation Loss'
    if run_intention:
        title += '\n' + run_intention
    ax.set_title(title, fontsize=14, fontweight='bold')
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)

    if freeze_encoder and unfreeze_after_epoch is not None and unfreeze_after_epoch > 0:
        ax.axvline(x=unfreeze_after_epoch, color='green', linestyle='-', linewidth=1.2)
        label_y = y_max - 0.03 * (y_max - y_min) if y_max > y_min else y_max
        ax.text(unfreeze_after_epoch, label_y, 'unfreeze', rotation=90, fontsize=9,
                color='green', va='bottom', ha='center')

    if early_stopping_patience is not None and early_stop_counter is not None and len(early_stop_counter) == len(epochs):
        ax2 = ax.twinx()
        ax2.bar(epochs, early_stop_counter, alpha=0.2, color='green', width=0.8)
        ax2.set_ylim(0, early_stopping_patience)
        ax2.set_ylabel('Early stop (patience)', fontsize=9, color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen', labelsize=8)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    info_lines = []
    if early_stopping_patience is not None and early_stop_counter is not None:
        early_note = "Early stop: val_loss"
        if early_stopping_min_delta is not None and early_stopping_min_delta > 0:
            early_note += f" (improvement >= {early_stopping_min_delta:.0e})"
        info_lines.append(early_note)
    if config_summary:
        info_lines.extend(s.strip() for s in config_summary.split("|") if s.strip())
    if num_train_tiles is not None or num_val_tiles is not None:
        tiles = f"tiles: train={num_train_tiles}" if num_train_tiles is not None else "tiles:"
        if num_val_tiles is not None:
            tiles += f" val={num_val_tiles}"
        info_lines.append(tiles)

    if training_start_datetime:
        info_lines.append(f"started: {training_start_datetime}")
    if training_duration_seconds is not None and training_duration_seconds >= 0:
        h = int(training_duration_seconds // 3600)
        m = int((training_duration_seconds % 3600) // 60)
        info_lines.append(f"duration: {h}h {m}m")

    # Wrap long lines so config text never stretches the figure when saving
    wrapped_lines = []
    for line in info_lines:
        if len(line) > 100:
            for i in range(0, len(line), 100):
                wrapped_lines.append(line[i : i + 100])
        else:
            wrapped_lines.append(line)
    info_lines = wrapped_lines

    if info_lines:
        fig.subplots_adjust(left=0.28, bottom=0.12, right=0.88, top=0.92)
        # Twin axis can keep a different position and blow up bbox_inches='tight'; sync to main axes
        if early_stopping_patience is not None and early_stop_counter is not None and len(early_stop_counter) == len(epochs):
            ax2.set_position(ax.get_position())
        fig.text(0.02, 0.04, '\n'.join(info_lines), fontsize=6, ha='left', va='bottom',
                 transform=fig.transFigure, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    else:
        plt.tight_layout()

    if output_path:
        # Fixed size (no bbox_inches='tight') so width doesn't explode with twin axes / long text
        fig.savefig(output_path, dpi=150)

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
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)

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
    loss_plot_options: Optional[Dict[str, Any]] = None,
    which_plots: Optional[Sequence[str]] = None,
) -> Dict[str, plt.Figure]:
    """
    Create training plots from metrics history.

    Args:
        metrics_history: Dictionary with keys like 'epochs', 'train_loss', 'val_loss',
                        'val_mae', 'val_iou', 'improvement_percent', optionally 'early_stop_counter'
        baseline_mae: Optional baseline MAE value
        output_dir: Optional directory to save figures
        loss_plot_options: Optional dict for loss plot: config_summary, num_train_tiles,
                          num_val_tiles, early_stopping_patience (early_stop_counter from metrics_history)
        which_plots: If set, only create these plot names ('loss', 'mae_comparison', 'iou',
                     'improvement_percent'). If None, create all. Use which_plots=['loss'] during
                     training loop to avoid building all four figures every epoch.

    Returns:
        Dictionary mapping plot names to matplotlib Figures
    """
    epochs = metrics_history.get('epochs', [])
    if not epochs:
        return {}

    figures = {}
    opts = loss_plot_options or {}
    want = set(which_plots) if which_plots is not None else None

    def _want(name: str) -> bool:
        return want is None or name in want

    # MAE comparison
    if _want('mae_comparison') and 'val_mae' in metrics_history:
        fig = plot_mae_comparison(
            epochs, metrics_history['val_mae'], baseline_mae
        )
        figures['mae_comparison'] = fig
        if output_dir:
            fig.savefig(output_dir / 'mae_comparison.png', dpi=150, bbox_inches='tight')

    # Loss plot
    if _want('loss') and 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        val_loss_list = metrics_history['val_loss']
        min_val = min(val_loss_list) if val_loss_list else None
        early_stop_counter = metrics_history.get('early_stop_counter')
        fig = plot_loss(
            epochs, metrics_history['train_loss'], val_loss_list,
            min_val_loss=min_val,
            early_stop_counter=early_stop_counter,
            early_stopping_patience=opts.get('early_stopping_patience'),
            early_stopping_min_delta=opts.get('early_stopping_min_delta'),
            config_summary=opts.get('config_summary'),
            num_train_tiles=opts.get('num_train_tiles'),
            num_val_tiles=opts.get('num_val_tiles'),
            freeze_encoder=opts.get('freeze_encoder'),
            unfreeze_after_epoch=opts.get('unfreeze_after_epoch'),
            training_start_datetime=opts.get('training_start_datetime'),
            training_duration_seconds=opts.get('training_duration_seconds'),
            run_intention=opts.get('run_intention'),
        )
        figures['loss'] = fig
        if output_dir:
            fig.savefig(output_dir / 'loss.png', dpi=150, bbox_inches='tight')

    # IoU plot
    if _want('iou') and 'val_iou' in metrics_history:
        fig = plot_iou(epochs, metrics_history['val_iou'])
        figures['iou'] = fig
        if output_dir:
            fig.savefig(output_dir / 'iou.png', dpi=150, bbox_inches='tight')

    # Improvement percentage (only when baseline was computed and list matches epochs)
    improvement_percent = metrics_history.get('improvement_percent', [])
    if _want('improvement_percent') and improvement_percent and len(improvement_percent) == len(epochs):
        fig = plot_improvement_percentage(epochs, improvement_percent)
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
    key_256 = "representative_tile_ids_synthetic_parenthesis_256"
    key_512 = "representative_tile_ids_synthetic_parenthesis_512"
    if mode == "synthetic_parenthesis_256" and key_256 in viz_config:
        return viz_config.get(key_256, [])
    if mode == "synthetic_parenthesis_512" and key_512 in viz_config:
        return viz_config.get(key_512, [])
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
    if not path.exists():
        return np.zeros((256, 256, 3), dtype=np.float32)  # placeholder
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float64)
    if rgb.size == 0:
        return np.zeros((256, 256, 3), dtype=np.float32)
    mx = float(np.max(rgb))
    if mx <= 0:
        pass
    elif mx <= 1.0:
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb.astype(np.float32)


def _load_raw_feature_channels_for_display(
    features_path: Path, features_base_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw R, G, B, DEM, Slope from feature GeoTIFF for display (no normalization)."""
    p = Path(features_path)
    path = (features_base_dir / p) if not p.is_absolute() else p
    empty = (
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
    )
    if not path.exists():
        return empty
    with rasterio.open(path) as src:
        if src.count < 5:
            return empty
        data = src.read([1, 2, 3, 4, 5])
    r = np.asarray(data[0], dtype=np.float64)
    g = np.asarray(data[1], dtype=np.float64)
    b = np.asarray(data[2], dtype=np.float64)
    dem = np.asarray(data[3], dtype=np.float64)
    slope = np.asarray(data[4], dtype=np.float64)
    return r, g, b, dem, slope


def _add_proximity_scale_and_extrema(
    ax: plt.Axes,
    data: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 20.0,
    colorbar_label: str = "proximity",
) -> None:
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.7, label=colorbar_label)
    flat = np.nan_to_num(data, nan=np.nanmean(data)).ravel()
    if flat.size == 0:
        return
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    min_idx = np.argmin(data)
    max_idx = np.argmax(data)
    min_rc = np.unravel_index(min_idx, data.shape)
    max_rc = np.unravel_index(max_idx, data.shape)
    ax.scatter(min_rc[1], min_rc[0], marker="x", s=100, c="black", linewidths=2, zorder=5)
    ax.scatter(max_rc[1], max_rc[0], marker="x", s=100, c="black", linewidths=2, zorder=5)
    ax.text(min_rc[1], min_rc[0] - 13, f"{min_val:.1f}", color="black", fontsize=8, ha="center", va="top", zorder=6)
    ax.text(max_rc[1], max_rc[0] + 13, f"{max_val:.1f}", color="black", fontsize=8, ha="center", va="bottom", zorder=6)


def _create_tile_prediction_figure(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    title: str,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

    rep_tiles = [tile_info]
    first_path = Path(features_dir) / tile_info.get("features_path", "").replace("\\", "/")
    if first_path.exists():
        with rasterio.open(first_path) as src:
            h, w = src.height, src.width
            if h == w:
                tile_size = int(h)
    model.eval()
    mode = (target_mode or "proximity").lower()
    dataset = TileDataset(
        rep_tiles,
        Path(features_dir),
        Path(targets_dir),
        normalization_stats,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
    )
    features, target = dataset[0]
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
    fp = tile_info.get("features_path", "")
    rgb = _load_rgb_for_display(Path(fp.replace("\\", "/")), Path(features_dir))
    tid = tile_info.get("tile_id", "tile_0")
    if mode == "binary":
        vmin, vmax = 0.0, 1.0
        colorbar_label = "lobe (0-1)"
        target_title = "Target (0/1)"
    else:
        vmin, vmax = 0.0, 20.0
        colorbar_label = "proximity"
        target_title = "Proximity (target)"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].set_title(target_title)
    _add_proximity_scale_and_extrema(axes[1], target_np, vmin, vmax, colorbar_label)
    axes[1].axis("off")
    axes[2].set_title("Prediction")
    _add_proximity_scale_and_extrema(axes[2], pred_np, vmin, vmax, colorbar_label)
    axes[2].axis("off")
    subtitle = f"{title}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  IoU: {iou:.4f}"
    if mode == "binary" and np.max(target_np) <= 0:
        subtitle += "  (background tile — target all 0)"
    fig.suptitle(subtitle, fontsize=10)
    plt.tight_layout()
    return fig


def show_best_predicted_tile(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    loss_value: float,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    tid = tile_info.get("tile_id", "?")
    title = f"Lowest-loss tile  |  {tid}  |  loss: {loss_value:.6f}"
    return _create_tile_prediction_figure(
        model,
        tile_info,
        features_dir,
        targets_dir,
        normalization_stats,
        device,
        tile_size,
        iou_threshold,
        title,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
    )


def show_highest_iou_tile(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    iou_value: float,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    tile_loss: Optional[float] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    tid = tile_info.get("tile_id", "?")
    title = f"Highest IoU tile  |  {tid}  |  IoU: {iou_value:.4f}"
    if tile_loss is not None:
        title += f"  |  loss: {tile_loss:.6f}"
    return _create_tile_prediction_figure(
        model,
        tile_info,
        features_dir,
        targets_dir,
        normalization_stats,
        device,
        tile_size,
        iou_threshold,
        title,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
    )


def create_prediction_tile_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    iou_threshold: float = 5.0,
    tile_size: int = 256,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
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
    mode = (target_mode or "proximity").lower()
    if mode == "binary":
        vmin, vmax = 0.0, 1.0
        target_title = "Target (0/1)"
    else:
        vmin, vmax = 0.0, 20.0
        target_title = "Proximity (target)"
    dataset = TileDataset(
        rep_tiles,
        features_dir,
        targets_dir,
        normalization_stats,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
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
        im1 = axes[1].imshow(target_np, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im1, ax=axes[1], shrink=0.7, label="Target" if mode == "binary" else "0–20")
        axes[1].set_title(target_title)
        axes[1].axis("off")
        im2 = axes[2].imshow(pred_np, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im2, ax=axes[2], shrink=0.7, label="Pred" if mode == "binary" else "0–20")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        title = f"Tile: {tid}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  IoU: {iou:.4f}"
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        figures[tid] = fig
    return figures


def _build_training_params_info_lines(plot_options: Optional[Dict[str, Any]] = None) -> List[str]:
    """Build info lines for training params (same style as advanced loss plot)."""
    if not plot_options:
        return []
    info_lines = []
    config_summary = plot_options.get("config_summary")
    if config_summary:
        info_lines.extend(s.strip() for s in config_summary.split("|") if s.strip())
    num_train = plot_options.get("num_train_tiles")
    num_val = plot_options.get("num_val_tiles")
    if num_train is not None or num_val is not None:
        tiles = f"tiles: train={num_train}" if num_train is not None else "tiles:"
        if num_val is not None:
            tiles += f" val={num_val}"
        info_lines.append(tiles)
    if plot_options.get("training_start_datetime"):
        info_lines.append(f"started: {plot_options['training_start_datetime']}")
    duration = plot_options.get("training_duration_seconds")
    if duration is not None and duration >= 0:
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        info_lines.append(f"duration: {h}h {m}m")
    wrapped = []
    for line in info_lines:
        if len(line) > 100:
            for i in range(0, len(line), 100):
                wrapped.append(line[i : i + 100])
        else:
            wrapped.append(line)
    return wrapped


def _channel_to_display(arr: np.ndarray) -> np.ndarray:
    """Normalize channel to [0,1] for display (min-max)."""
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    mn, mx = np.min(a), np.max(a)
    if mx - mn > 1e-12:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0, 1).astype(np.float32)


def _segment_boundary_mask(seg: np.ndarray, nodata: float) -> np.ndarray:
    """True where pixel differs from any 4-neighbor (segment boundary). Nodata excluded."""
    out = np.zeros(seg.shape, dtype=bool)
    valid = seg != nodata
    if not np.any(valid):
        return out
    s = np.asarray(seg, dtype=np.float64)
    out[:-1, :] |= valid[:-1, :] & valid[1:, :] & (s[:-1, :] != s[1:, :])
    out[1:, :]  |= valid[1:, :] & valid[:-1, :] & (s[1:, :] != s[:-1, :])
    out[:, :-1] |= valid[:, :-1] & valid[:, 1:] & (s[:, :-1] != s[:, 1:])
    out[:, 1:]  |= valid[:, 1:] & valid[:, :-1] & (s[:, 1:] != s[:, :-1])
    return out


def _load_segmentation_for_display(
    segmentation_base_dir: Path, tile_id: str,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Load segmentation tile for display. Returns (display_array, boundary_mask, n_unique) or None.
    Display array: segment IDs mapped to 0-1 discretely. boundary_mask: True at segment edges."""
    seg_path = segmentation_base_dir / f"{tile_id}.tif"
    if not seg_path.exists():
        return None
    with rasterio.open(seg_path) as src:
        seg = src.read(1)
        nd = float(src.nodata if src.nodata is not None else -9999.0)
    seg = np.asarray(seg, dtype=np.float64)
    valid = seg != nd
    out = np.zeros_like(seg, dtype=np.float32)
    n_uniq = 0
    if np.any(valid):
        v = seg[valid].astype(np.int64)
        n_uniq = len(np.unique(v))
        out[valid] = (v % 64) / 63.0
    boundaries = _segment_boundary_mask(seg, nd)
    return (np.clip(out, 0, 1).astype(np.float32), boundaries, n_uniq)


def create_representative_tiles_channel_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    iou_threshold: float,
    tile_size: int,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    plot_options: Optional[Dict[str, Any]] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> Dict[str, plt.Figure]:
    """
    For each representative tile: one figure with RGB, DEM, Slope, [Segmentation], [SlopeStripes],
    Target, Prediction. Layout: 2 rows x 4 columns; bottom-left cell empty for text box.
    """
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_iou

    if not rep_tiles:
        return {}
    first_path = Path(features_dir) / rep_tiles[0].get("features_path", "").replace("\\", "/")
    if first_path.exists():
        with rasterio.open(first_path) as src:
            h, w = src.height, src.width
            if h == w:
                tile_size = int(h)
    model.eval()
    mode = (target_mode or "proximity").lower()
    vmin, vmax = (0.0, 1.0) if mode == "binary" else (0.0, 20.0)
    dataset = TileDataset(
        rep_tiles,
        features_dir,
        targets_dir,
        normalization_stats,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
    )
    info_lines = _build_training_params_info_lines(plot_options)
    figures = {}
    n_rows, n_cols = 2, 4
    ax_slot_order = [0, 1, 2, 3, 5, 6, 7]
    for i, tile_info in enumerate(rep_tiles):
        features, target = dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(features_batch)
        pred_np = np.squeeze(pred.cpu().numpy())
        target_np = target.squeeze().numpy()
        fp = tile_info.get("features_path", "").replace("\\", "/")
        tid = tile_info.get("tile_id", f"tile_{i}")
        raw_r, raw_g, raw_b, raw_dem, raw_slope = _load_raw_feature_channels_for_display(
            Path(fp), Path(features_dir)
        )
        rgb = _load_rgb_for_display(Path(fp), Path(features_dir))
        panels = []
        panels.append(("RGB", rgb, "rgb"))
        dem_disp = _channel_to_display(raw_dem)
        mn_dem, mx_dem = float(np.nanmin(raw_dem)), float(np.nanmax(raw_dem))
        panels.append(("DEM", (dem_disp, f"{mn_dem:.1f}–{mx_dem:.1f}"), "gray_label"))
        slope_disp = _channel_to_display(raw_slope)
        mn_sl, mx_sl = float(np.nanmin(raw_slope)), float(np.nanmax(raw_slope))
        panels.append(("Slope", (slope_disp, f"{mn_sl:.1f}–{mx_sl:.1f}"), "gray_label"))
        seg_used = segmentation_base_dir is not None
        seg_result = _load_segmentation_for_display(segmentation_base_dir, tid) if seg_used else None
        if seg_result is not None:
            seg_arr, seg_boundaries, n_seg = seg_result
            seg_title = f"Segmentation ({n_seg} segments)"
            panels.append((seg_title, ("segmentation_contours", _channel_to_display(seg_arr), seg_boundaries), "segmentation_contours"))
        else:
            h, w = raw_dem.shape
            placeholder = np.zeros((h, w), dtype=np.float32)
            seg_title = "Segmentation (disabled)" if not seg_used else "Segmentation (no file)"
            panels.append((seg_title, _channel_to_display(placeholder), "gray"))
        if slope_stripes_base_dir is not None:
            stripe = features[-1].numpy()
            panels.append(("SlopeStripes", _channel_to_display(stripe), "gray"))
        panels.append(("Target", target_np, "viridis"))
        panels.append(("Prediction", pred_np, "viridis"))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes_flat = np.atleast_2d(axes).ravel()
        for slot_idx, (title, data, kind) in zip(ax_slot_order, panels):
            ax = axes_flat[slot_idx]
            if kind == "rgb":
                ax.imshow(data)
                ax.set_title(title)
            elif kind == "segmentation_contours":
                _, gray_arr, boundary_mask = data
                ax.imshow(gray_arr, cmap="gray")
                overlay = np.ma.masked_where(~boundary_mask, np.ones_like(boundary_mask, dtype=np.float32))
                white = mcolors.ListedColormap([[1, 1, 1, 1]])
                ax.imshow(overlay, cmap=white, vmin=0, vmax=1, alpha=0.85)
                ax.set_title(title)
            elif kind == "gray":
                im = ax.imshow(data, cmap="gray")
                plt.colorbar(im, ax=ax, shrink=0.6)
                ax.set_title(title)
            elif kind == "gray_label":
                im_arr, label = data
                im = ax.imshow(im_arr, cmap="gray")
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar.set_label(label)
                ax.set_title(title)
            else:
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
                plt.colorbar(im, ax=ax, shrink=0.6, label="0–1" if mode == "binary" else "0–20")
                ax.set_title(title)
            ax.axis("off")
        axes_flat[4].axis("off")
        for j in range(len(panels), len(ax_slot_order)):
            axes_flat[ax_slot_order[j]].axis("off")
        pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)
        mae = compute_mae(pred_t, target_t)
        iou = compute_iou(pred_t, target_t, threshold=iou_threshold)
        title = f"Representative tile: {tid}  |  MAE: {mae:.4f}  IoU: {iou:.4f}"
        if plot_options and plot_options.get("run_intention"):
            title += "\n" + plot_options["run_intention"]
        fig.suptitle(title, fontsize=11)
        if info_lines:
            fig.subplots_adjust(left=0.02, bottom=0.18, right=0.98, top=0.92)
            fig.text(
                0.02, 0.02, "\n".join(info_lines), fontsize=6, ha="left", va="bottom",
                transform=fig.transFigure, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="lightgray"),
            )
        else:
            plt.tight_layout()
        figures[tid] = fig
    return figures
