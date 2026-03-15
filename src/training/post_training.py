"""
Post-training: log final metrics, print run end, load best checkpoint, visualizations, save MLflow model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.training.training_loop import TrainingLoopResult
from src.training.visualization import (
    create_training_plots,
    create_prediction_tile_figures,
    create_representative_tiles_channel_figures,
    get_representative_tile_ids_for_viz,
    resolve_representative_tiles,
    show_best_predicted_tile,
    show_highest_iou_tile,
)
from src.utils.mlflow_utils import save_model

logger = logging.getLogger(__name__)


def log_final_metrics_and_trial_attrs(
    best_val_loss: float,
    best_val_mae: float,
    best_val_iou: float,
    trial: Optional[Any],
) -> None:
    import mlflow
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_metric("best_val_mae", best_val_mae)
    mlflow.log_metric("best_val_iou", best_val_iou)
    if trial is not None and hasattr(trial, "set_user_attr"):
        try:
            trial.set_user_attr("best_val_loss", float(best_val_loss))
            trial.set_user_attr("best_val_mae", float(best_val_mae))
            trial.set_user_attr("best_val_iou", float(best_val_iou))
        except Exception:
            pass


def print_mlflow_run_end(run_name: str, best_val_loss: float, trial: Optional[Any]) -> None:
    import mlflow
    active = mlflow.active_run()
    if active is None:
        return
    exp_id = active.info.experiment_id
    run_id = active.info.run_id
    tracking_uri = mlflow.get_tracking_uri()
    if trial is not None:
        print(
            f"[MLFLOW][OPTUNA][DONE] trial={getattr(trial, 'number', '?')} "
            f"experiment_id={exp_id} run_id={run_id} run_name={run_name} "
            f"best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
    else:
        print(
            f"[MLFLOW][DONE] experiment_id={exp_id} run_id={run_id} run_name={run_name} "
            f"best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
    print(f"[MLFLOW] tracking_uri={tracking_uri}", flush=True)


def load_best_checkpoint(
    model: torch.nn.Module,
    best_model_path: Path,
    device: torch.device,
) -> None:
    if not best_model_path.exists():
        return
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded best checkpoint for plots and MLflow")


def run_post_training_visualization(
    config: dict,
    model: torch.nn.Module,
    result: TrainingLoopResult,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    target_mode: str,
    binary_threshold: float,
    segmentation_dir: Optional[Path],
    slope_stripes_channel_dir: Optional[Path],
    use_rgb: bool,
    use_dem: bool,
    use_slope: bool,
    all_tiles: List[dict],
    path_key: str,
    loss_plot_path: Optional[Path],
    loss_plot_options: dict,
    elapsed_seconds: float,
) -> None:
    import matplotlib.pyplot as plt
    import mlflow

    logger.info("=== Creating Training Plots ===")
    loss_plot_options["training_duration_seconds"] = elapsed_seconds
    figures = create_training_plots(
        result.metrics_history, result.baseline_mae, loss_plot_options=loss_plot_options
    )
    for plot_name, fig in figures.items():
        mlflow.log_figure(fig, f"plots/{plot_name}.png")
        if plot_name == "loss" and loss_plot_path is not None:
            fig.savefig(loss_plot_path, dpi=150)
        plt.close(fig)
        logger.info("Logged %s plot to MLflow", plot_name)

    if result.best_tile_info_so_far is not None:
        logger.info("=== Creating best predicted tile figure ===")
        fig = show_best_predicted_tile(
            model,
            result.best_tile_info_so_far,
            features_dir,
            targets_dir,
            normalization_stats,
            device,
            tile_size,
            iou_threshold,
            result.best_tile_loss_so_far,
            target_mode=target_mode,
            binary_threshold=binary_threshold,
            segmentation_base_dir=segmentation_dir,
            slope_stripes_base_dir=slope_stripes_channel_dir,
            use_rgb=use_rgb,
            use_dem=use_dem,
            use_slope=use_slope,
        )
        mlflow.log_figure(fig, "plots/best_predicted_tile.png")
        if loss_plot_path is not None:
            fig.savefig(loss_plot_path.parent / "best_predicted_tile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    if result.best_iou_tile_info_so_far is not None:
        logger.info("=== Creating best IoU tile figure ===")
        fig = show_highest_iou_tile(
            model,
            result.best_iou_tile_info_so_far,
            features_dir,
            targets_dir,
            normalization_stats,
            device,
            tile_size,
            iou_threshold,
            result.best_iou_so_far,
            target_mode=target_mode,
            binary_threshold=binary_threshold,
            segmentation_base_dir=segmentation_dir,
            slope_stripes_base_dir=slope_stripes_channel_dir,
            tile_loss=result.best_iou_tile_loss_so_far,
            use_rgb=use_rgb,
            use_dem=use_dem,
            use_slope=use_slope,
        )
        mlflow.log_figure(fig, "plots/best_iou_tile.png")
        if loss_plot_path is not None:
            fig.savefig(loss_plot_path.parent / "best_iou_tile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    viz_config = config.get("visualization", {})
    rep_tile_ids = get_representative_tile_ids_for_viz(viz_config, path_key, tile_size)
    rep_tiles = resolve_representative_tiles(all_tiles, rep_tile_ids) if rep_tile_ids else []
    fallback_n = int(viz_config.get("prediction_tiles_fallback_n", 0))
    if not rep_tiles and fallback_n > 0 and all_tiles:
        rep_tiles = all_tiles[:fallback_n]
        logger.info(
            "Using first %d tiles for prediction viz (configured IDs did not match or none set)",
            len(rep_tiles),
        )
    if rep_tiles:
        logger.info("=== Creating prediction tile visualizations ===")
        pred_figures = create_prediction_tile_figures(
            model,
            rep_tiles,
            features_dir,
            targets_dir,
            normalization_stats,
            device,
            iou_threshold=iou_threshold,
            tile_size=tile_size,
            target_mode=target_mode,
            binary_threshold=binary_threshold,
            segmentation_base_dir=segmentation_dir,
            slope_stripes_base_dir=slope_stripes_channel_dir,
            use_rgb=use_rgb,
            use_dem=use_dem,
            use_slope=use_slope,
        )
        for tid, fig in pred_figures.items():
            mlflow.log_figure(fig, f"prediction_tiles/{tid}.png")
            plt.close(fig)
            logger.info("Logged prediction tile: %s", tid)
        logger.info("=== Creating representative tiles channel visualizations ===")
        channel_figures = create_representative_tiles_channel_figures(
            model,
            rep_tiles,
            features_dir,
            targets_dir,
            normalization_stats,
            device,
            iou_threshold=iou_threshold,
            tile_size=tile_size,
            target_mode=target_mode,
            binary_threshold=binary_threshold,
            segmentation_base_dir=segmentation_dir,
            slope_stripes_base_dir=slope_stripes_channel_dir,
            plot_options=loss_plot_options,
            use_rgb=use_rgb,
            use_dem=use_dem,
            use_slope=use_slope,
        )
        for tid, fig in channel_figures.items():
            mlflow.log_figure(fig, f"representative_channels/{tid}.png")
            plt.close(fig)
            logger.info("Logged representative channels: %s", tid)
    elif rep_tile_ids:
        logger.warning(
            "representative_tile_ids configured but no matching tiles found and fallback_n=0; "
            "check tile IDs match filtered_tiles.json or set prediction_tiles_fallback_n > 0"
        )


def save_mlflow_model_if_enabled(
    model: torch.nn.Module,
    mlflow_config: dict,
    trial: Optional[Any],
) -> None:
    if not mlflow_config.get("log_model", True) or trial is not None:
        return
    model_size_mb = save_model(model, "model")
    if model_size_mb > 0:
        logger.info("Model saved to MLflow (size: %.2f MB)", model_size_mb)
