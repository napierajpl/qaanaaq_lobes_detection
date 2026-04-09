"""Visualization utilities for training metrics.

This module re-exports all public symbols from the split sub-modules
for backward compatibility. New code should import directly from:
  - src.training.loss_plots
  - src.training.prediction_tiles
  - src.training.channel_figures
"""

from src.training.loss_plots import (  # noqa: F401
    _calculate_smart_ylim,
    _xlim_epochs,
    _plot_learning_rate_axis,
    plot_mae_comparison,
    plot_loss_simple,
    plot_loss,
    plot_iou,
    plot_improvement_percentage,
    create_training_plots,
)
from src.training.prediction_tiles import (  # noqa: F401
    get_representative_tile_ids_for_viz,
    _tile_id_to_index,
    resolve_representative_tiles,
    show_tile_prediction,
    show_best_predicted_tile,
    show_highest_iou_tile,
    create_prediction_tile_figures,
    _load_rgb_for_display,
    _add_proximity_scale_and_extrema,
)
from src.training.channel_figures import (  # noqa: F401
    create_representative_tiles_channel_figures,
    _build_training_params_info_lines,
    _channel_to_display,
    _segment_boundary_mask,
    _load_segmentation_for_display,
)
