"""
Metric plotting utilities: loss, MAE, IoU, improvement percentage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _calculate_smart_ylim(
    values: List[float],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    padding: float = 0.1,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    values_array = np.array(values)
    y_min = np.percentile(values_array, lower_percentile)
    y_max = np.percentile(values_array, upper_percentile)
    y_range = y_max - y_min
    if y_range > 0:
        y_min = max(0, y_min - padding * y_range)
        y_max = y_max + padding * y_range
    else:
        y_min = max(0, y_min - abs(y_min) * padding)
        y_max = y_max + abs(y_max) * padding
    return float(y_min), float(y_max)


def _xlim_epochs(epochs: List[int]) -> Tuple[float, float]:
    if not epochs:
        return 0.0, 1.0
    lo, hi = min(epochs), max(epochs)
    if lo == hi:
        return float(lo) - 0.5, float(hi) + 0.5
    return float(lo), float(hi)


def _plot_learning_rate_axis(
    ax: plt.Axes,
    epochs: List[int],
    learning_rate: Optional[List[float]],
) -> None:
    lr_values = learning_rate or []
    if lr_values:
        n = min(len(epochs), len(lr_values))
        x = epochs[:n]
        y = lr_values[:n]
        ax.plot(x, y, color='purple', linewidth=1.6, marker='.', markersize=3, label='Learning Rate')
        lr_min, lr_max = _calculate_smart_ylim(y, lower_percentile=0.0, upper_percentile=100.0, padding=0.05)
        if lr_min == lr_max:
            lr_min = max(0.0, lr_min * 0.95)
            lr_max = lr_max * 1.05 if lr_max > 0 else 1.0
        ax.set_ylim(bottom=lr_min, top=lr_max)
        ax.legend(loc='best', fontsize=8)
    ax.set_ylabel('LR', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=12)
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)
    ax.grid(True, alpha=0.3)


def plot_mae_comparison(
    epochs: List[int],
    model_mae: List[float],
    baseline_mae: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, model_mae, 'b-', linewidth=2, label='Model MAE', marker='o', markersize=4)
    if baseline_mae is not None:
        ax.axhline(y=baseline_mae, color='r', linestyle='--', linewidth=2, label=f'Baseline MAE ({baseline_mae:.4f})')
    y_min, y_max = _calculate_smart_ylim(model_mae)
    if baseline_mae is not None:
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
    learning_rate: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [5, 1]},
    )
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    all_loss_values = train_loss + val_loss
    y_min, y_max = _calculate_smart_ylim(all_loss_values)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    x_lo, x_hi = _xlim_epochs(epochs)
    ax.set_xlim(left=x_lo, right=x_hi)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    _plot_learning_rate_axis(ax_lr, epochs, learning_rate)
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
    learning_rate: Optional[List[float]] = None,
) -> plt.Figure:
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [5, 1]},
    )
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
    _plot_learning_rate_axis(ax_lr, epochs, learning_rate)
    info_lines = _build_loss_plot_info_lines(
        early_stopping_patience, early_stop_counter, early_stopping_min_delta,
        config_summary, num_train_tiles, num_val_tiles,
        training_start_datetime, training_duration_seconds,
    )
    if info_lines:
        fig.subplots_adjust(left=0.28, bottom=0.12, right=0.88, top=0.92)
        if early_stopping_patience is not None and early_stop_counter is not None and len(early_stop_counter) == len(epochs):
            ax2.set_position(ax.get_position())
        fig.text(0.02, 0.04, '\n'.join(info_lines), fontsize=6, ha='left', va='bottom',
                 transform=fig.transFigure, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    else:
        plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def _build_loss_plot_info_lines(
    early_stopping_patience, early_stop_counter, early_stopping_min_delta,
    config_summary, num_train_tiles, num_val_tiles,
    training_start_datetime, training_duration_seconds,
) -> List[str]:
    info_lines: List[str] = []
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
    return _wrap_long_lines(info_lines)


def _wrap_long_lines(lines: List[str], max_len: int = 100) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        if len(line) > max_len:
            for i in range(0, len(line), max_len):
                wrapped.append(line[i : i + max_len])
        else:
            wrapped.append(line)
    return wrapped


def plot_iou(
    epochs: List[int],
    val_iou: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_iou, 'g-', linewidth=2, label='Val IoU', marker='o', markersize=4)
    y_min, y_max = _calculate_smart_ylim(val_iou)
    y_min = max(0, y_min)
    y_max = min(1.0, max(1.0, y_max))
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
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['g' if p >= 0 else 'r' for p in improvement_percent]
    ax.bar(epochs, improvement_percent, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
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
    epochs = metrics_history.get('epochs', [])
    if not epochs:
        return {}
    figures = {}
    opts = loss_plot_options or {}
    want = set(which_plots) if which_plots is not None else None

    def _want(name: str) -> bool:
        return want is None or name in want

    if _want('mae_comparison') and 'val_mae' in metrics_history:
        fig = plot_mae_comparison(epochs, metrics_history['val_mae'], baseline_mae)
        figures['mae_comparison'] = fig
        if output_dir:
            fig.savefig(output_dir / 'mae_comparison.png', dpi=150, bbox_inches='tight')
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
            learning_rate=metrics_history.get('learning_rate'),
        )
        figures['loss'] = fig
        if output_dir:
            fig.savefig(output_dir / 'loss.png', dpi=150, bbox_inches='tight')
    if _want('iou') and 'val_iou' in metrics_history:
        fig = plot_iou(epochs, metrics_history['val_iou'])
        figures['iou'] = fig
        if output_dir:
            fig.savefig(output_dir / 'iou.png', dpi=150, bbox_inches='tight')
    improvement_percent = metrics_history.get('improvement_percent', [])
    if _want('improvement_percent') and improvement_percent and len(improvement_percent) == len(epochs):
        fig = plot_improvement_percentage(epochs, improvement_percent)
        figures['improvement_percent'] = fig
        if output_dir:
            fig.savefig(output_dir / 'improvement_percent.png', dpi=150, bbox_inches='tight')
    return figures
