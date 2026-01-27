"""
Visualization utilities for training metrics.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


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
