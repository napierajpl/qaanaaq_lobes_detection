#!/usr/bin/env python3
"""
Train CNN model for lobe detection.
"""

import logging
import sys
import warnings
import yaml
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
except ImportError:
    optuna = None

import numpy as np
import torch
import torch.optim as optim
import mlflow
import matplotlib.pyplot as plt

from src.models.factory import create_model
from src.models.losses import (
    SmoothL1Loss,
    WeightedSmoothL1Loss,
    DiceLoss,
    IoULoss,
    SoftIoULoss,
    EncouragementLoss,
    FocalLoss,
    CombinedLoss,
)
from src.training.dataloader import (
    load_filtered_tiles,
    create_data_splits,
    create_dataloaders,
)
from src.training.trainer import train_one_epoch, validate, save_checkpoint
from src.training.visualization import create_training_plots
from src.preprocessing.normalization import compute_statistics
from src.utils.mlflow_utils import setup_mlflow_experiment, log_training_config, save_model, log_metrics
from src.utils.path_utils import get_project_root, resolve_path
from src.map_overlays.tile_registry import TileRegistry
from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce log noise from known, non-actionable warnings
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The verbose parameter is deprecated\..*",
    category=UserWarning,
)


def train_model_with_config(
    config: dict,
    mode: str = "production",
    trial: Optional[object] = None,
    run_name: Optional[str] = None,
) -> float:
    """
    Train model with given configuration.

    This function can be called directly or used by Optuna for hyperparameter tuning.

    Args:
        config: Training configuration dictionary
        mode: "dev" or "production"
        trial: Optional Optuna trial object (for pruning)
        run_name: Optional MLflow run name

    Returns:
        Best validation loss
    """
    from pathlib import Path

    project_root = get_project_root(Path(__file__))

    # Determine paths
    paths = config["paths"][mode]

    filtered_tiles_path = resolve_path(Path(paths["filtered_tiles"]), project_root)
    features_dir = resolve_path(Path(paths["features_dir"]), project_root)
    targets_dir = resolve_path(Path(paths["targets_dir"]), project_root)
    models_dir = resolve_path(Path(paths["models_dir"]), project_root)

    logger.info("=== Training Configuration ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Filtered tiles: {filtered_tiles_path}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Targets dir: {targets_dir}")
    logger.info(f"Models dir: {models_dir}")

    # Load filtered tiles
    logger.info("Loading filtered tiles...")
    all_tiles = load_filtered_tiles(filtered_tiles_path)
    logger.info(f"Total tiles: {len(all_tiles)}")

    # Split into train/val/test
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    test_split = config["data"]["test_split"]

    # Validate splits
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) >= 1e-6:
        raise ValueError(
            f"Data splits must sum to 1.0, got {split_sum:.6f}. "
            f"train={train_split}, val={val_split}, test={test_split}"
        )

    train_tiles, val_tiles, test_tiles = create_data_splits(
        all_tiles,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )
    logger.info(f"Train: {len(train_tiles)}, Val: {len(val_tiles)}, Test: {len(test_tiles)}")

    # Compute normalization statistics from training set
    logger.info("Computing normalization statistics...")
    train_feature_paths = [features_dir / tile["features_path"] for tile in train_tiles]
    normalization_stats = compute_statistics(train_feature_paths)
    dem_mean = normalization_stats.get('dem', {}).get('mean', 'N/A')
    dem_std = normalization_stats.get('dem', {}).get('std', 'N/A')
    slope_mean = normalization_stats.get('slope', {}).get('mean', 'N/A')
    slope_std = normalization_stats.get('slope', {}).get('std', 'N/A')

    dem_mean_str = f"{dem_mean:.2f}" if isinstance(dem_mean, (int, float)) else str(dem_mean)
    dem_std_str = f"{dem_std:.2f}" if isinstance(dem_std, (int, float)) else str(dem_std)
    slope_mean_str = f"{slope_mean:.2f}" if isinstance(slope_mean, (int, float)) else str(slope_mean)
    slope_std_str = f"{slope_std:.2f}" if isinstance(slope_std, (int, float)) else str(slope_std)

    logger.info(f"DEM stats: mean={dem_mean_str}, std={dem_std_str}")
    logger.info(f"Slope stats: mean={slope_mean_str}, std={slope_std_str}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_tiles,
        val_tiles,
        features_dir,
        targets_dir,
        normalization_stats,
        batch_size=config["training"]["batch_size"],
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model using factory
    logger.info("Creating model...")
    model = create_model(config["model"]).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {trainable_params:,})")

    # Log architecture info
    architecture = config["model"].get("architecture", "unet")
    logger.info(f"Architecture: {architecture}")
    if architecture == "satlaspretrain_unet":
        encoder_config = config["model"].get("encoder", {})
        logger.info(f"Encoder: {encoder_config.get('name', 'unknown')} "
                   f"(pretrained={encoder_config.get('pretrained', False)}, "
                   f"frozen={encoder_config.get('freeze_encoder', False)})")

    # Setup loss and optimizer
    iou_threshold = config["training"].get("iou_threshold", 5.0)

    loss_function = config["training"]["loss_function"]
    if loss_function == "smooth_l1":
        criterion = SmoothL1Loss()
        logger.info("Using SmoothL1Loss")
    elif loss_function == "weighted_smooth_l1":
        lobe_weight = config["training"].get("lobe_weight", 5.0)
        lobe_threshold = config["training"].get("iou_threshold", 5.0)
        criterion = WeightedSmoothL1Loss(lobe_weight=lobe_weight, lobe_threshold=lobe_threshold)
        logger.info(f"Using WeightedSmoothL1Loss with lobe_weight={lobe_weight}, lobe_threshold={lobe_threshold}")
    elif loss_function == "dice":
        criterion = DiceLoss(threshold=iou_threshold)
        logger.info(f"Using DiceLoss with threshold={iou_threshold}")
    elif loss_function == "iou":
        criterion = IoULoss(threshold=iou_threshold)
        logger.info(f"Using IoULoss with threshold={iou_threshold}")
    elif loss_function == "soft_iou":
        criterion = SoftIoULoss(threshold=iou_threshold)
        logger.info(f"Using SoftIoULoss with threshold={iou_threshold}")
    elif loss_function == "encouragement":
        encouragement_weight = config["training"].get("encouragement_weight", 2.0)
        criterion = EncouragementLoss(
            lobe_threshold=iou_threshold,
            encouragement_weight=encouragement_weight,
        )
        logger.info(f"Using EncouragementLoss with threshold={iou_threshold}, encouragement_weight={encouragement_weight}")
    elif loss_function == "focal":
        alpha = config["training"].get("focal_alpha", 0.25)
        gamma = config["training"].get("focal_gamma", 2.0)
        criterion = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            lobe_threshold=iou_threshold,
        )
        logger.info(f"Using FocalLoss with alpha={alpha}, gamma={gamma}, lobe_threshold={iou_threshold}")
    elif loss_function == "combined":
        iou_weight = config["training"].get("iou_weight", 0.5)
        regression_weight = config["training"].get("regression_weight", 0.5)
        lobe_weight = config["training"].get("lobe_weight", 5.0)
        use_soft_iou = config["training"].get("use_soft_iou", False)
        criterion = CombinedLoss(
            iou_weight=iou_weight,
            regression_weight=regression_weight,
            iou_threshold=iou_threshold,
            lobe_weight=lobe_weight,
            lobe_threshold=iou_threshold,
            use_soft_iou=use_soft_iou,
        )
        soft_iou_str = " (with soft IoU)" if use_soft_iou else ""
        logger.info(f"Using CombinedLoss (IoU + Weighted Smooth L1){soft_iou_str} with "
                   f"iou_weight={iou_weight}, regression_weight={regression_weight}, threshold={iou_threshold}")
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Setup learning rate scheduler
    lr_scheduler = None
    lr_scheduler_config = config["training"].get("lr_scheduler")
    if lr_scheduler_config:
        if lr_scheduler_config == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config["training"].get("lr_scheduler_factor", 0.5),
                patience=config["training"].get("lr_scheduler_patience", 10),
                min_lr=config["training"].get("lr_scheduler_min_lr", 1e-6),
            )
            patience = config['training'].get('lr_scheduler_patience', 10)
            factor = config['training'].get('lr_scheduler_factor', 0.5)
            logger.info(f"Using ReduceLROnPlateau scheduler (patience={patience}, factor={factor})")
        else:
            logger.warning(f"Unknown LR scheduler '{lr_scheduler_config}', ignoring")

    # Setup early stopping
    early_stopping_patience = config["training"].get("early_stopping_patience")
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    early_stopping_counter = 0
    best_val_loss_for_early_stop = float("inf")

    if early_stopping_patience:
        logger.info(f"Early stopping enabled (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})")

    # Gradient clipping
    max_grad_norm = config["training"].get("max_grad_norm")
    if max_grad_norm:
        logger.info(f"Gradient clipping enabled (max_norm={max_grad_norm})")

    # Setup MLflow
    mlflow_config = config["mlflow"]
    setup_mlflow_experiment(mlflow_config["experiment_name"], mlflow_config.get("tracking_uri"))

    if run_name is None:
        run_name = f"unet_baseline_{mode}"

    with mlflow.start_run(run_name=run_name):

        # Log config
        log_training_config(config)
        mlflow.log_param("mode", mode)
        mlflow.log_param("num_params", num_params)
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("num_train_tiles", len(train_tiles))
        mlflow.log_param("num_val_tiles", len(val_tiles))

        # Log Optuna trial info if available
        if trial is not None:
            mlflow.set_tag("optuna_trial", str(trial.number))
            mlflow.set_tag("optuna_study", "lobe_detection_hp_tuning")
            # Log all trial parameters
            for key, value in trial.params.items():
                mlflow.log_param(f"optuna.{key}", value)

        # Capture MLflow IDs + key “compatibility” metadata on the Optuna trial itself
        # so `tune_hyperparameters.py` can export a rich CSV later.
        active = mlflow.active_run()
        if trial is not None and active is not None and hasattr(trial, "set_user_attr"):
            try:
                trial.set_user_attr("mlflow_experiment_id", active.info.experiment_id)
                trial.set_user_attr("mlflow_run_id", active.info.run_id)
                trial.set_user_attr("mlflow_run_name", run_name)
                trial.set_user_attr("mlflow_tracking_uri", mlflow.get_tracking_uri())
                trial.set_user_attr("mode", mode)
                trial.set_user_attr("model_architecture", architecture)
                trial.set_user_attr("model_in_channels", int(config["model"].get("in_channels", 5)))
                trial.set_user_attr("model_out_channels", int(config["model"].get("out_channels", 1)))
                trial.set_user_attr("training_iou_threshold", float(iou_threshold))
                trial.set_user_attr("data_normalize_rgb", bool(config["data"].get("normalize_rgb", True)))
                trial.set_user_attr("data_standardize_dem", bool(config["data"].get("standardize_dem", True)))
                trial.set_user_attr("data_standardize_slope", bool(config["data"].get("standardize_slope", True)))
                trial.set_user_attr("filtered_tiles_path", str(filtered_tiles_path))
                trial.set_user_attr("features_dir", str(features_dir))
                trial.set_user_attr("targets_dir", str(targets_dir))
                targets_str = str(targets_dir)
                if "proximity20" in targets_str:
                    trial.set_user_attr("proximity_token", "proximity20")
                elif "proximity10" in targets_str:
                    trial.set_user_attr("proximity_token", "proximity10")
                else:
                    trial.set_user_attr("proximity_token", "unknown")
            except Exception:
                # Never fail training because of metadata capture
                pass

        # Log architecture info
        mlflow.log_param("model.architecture", architecture)
        if architecture == "satlaspretrain_unet":
            encoder_config = config["model"].get("encoder", {})
            mlflow.log_param("model.encoder_name", encoder_config.get("name", "unknown"))
            mlflow.log_param("model.encoder_pretrained", encoder_config.get("pretrained", False))
            mlflow.log_param("model.encoder_frozen", encoder_config.get("freeze_encoder", False))
            mlflow.log_param("model.encoder_unfreeze_epoch", encoder_config.get("unfreeze_after_epoch", 0))

        # Detect and log proximity map parameters from target tiles
        proximity_max_value = None
        proximity_max_distance = None

        targets_path_str = str(targets_dir)
        if "proximity10px" in targets_path_str or "proximity10" in targets_path_str:
            proximity_max_value = 10
            proximity_max_distance = 10
        elif "proximity20px" in targets_path_str or "proximity20" in targets_path_str:
            proximity_max_value = 20
            proximity_max_distance = 20
        else:
            try:
                import rasterio
                sample_tiles = val_tiles[:min(5, len(val_tiles))]
                max_values = []
                for tile_info in sample_tiles:
                    tile_path = targets_dir / tile_info["targets_path"]
                    if tile_path.exists():
                        with rasterio.open(tile_path) as raster_src:
                            data = raster_src.read(1)
                            max_values.append(data.max())

                if max_values:
                    detected_max = int(max(max_values))
                    if detected_max <= 10:
                        proximity_max_value = 10
                        proximity_max_distance = 10
                    elif detected_max <= 20:
                        proximity_max_value = 20
                        proximity_max_distance = 20
                    else:
                        proximity_max_value = detected_max
                        proximity_max_distance = detected_max
            except (rasterio.RasterioIOError, ValueError, KeyError) as e:
                logger.warning(f"Could not detect proximity map parameters: {e}")

        if proximity_max_value is not None:
            mlflow.log_param("data.proximity_max_value", proximity_max_value)
            mlflow.log_param("data.proximity_max_distance", proximity_max_distance)
            logger.info(f"Detected proximity map: max_value={proximity_max_value}, max_distance={proximity_max_distance}")

        # Training loop
        best_val_loss = float("inf")
        best_model_path = models_dir / "best_model.pt"
        iou_threshold = config["training"].get("iou_threshold", 5.0)

        # Track metrics across epochs for visualization
        metrics_history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_iou": [],
            "improvement_percent": [],
        }
        baseline_mae = None

        logger.info("=== Starting Training ===")
        logger.info(f"IoU threshold: {iou_threshold}")

        # Encoder unfreezing configuration
        encoder_config = config["model"].get("encoder", {})
        unfreeze_after_epoch = encoder_config.get("unfreeze_after_epoch", 0)
        encoder_unfrozen = False

        for epoch in range(1, config["training"]["num_epochs"] + 1):
            # Unfreeze encoder if specified epoch reached
            if (unfreeze_after_epoch > 0 and
                epoch == unfreeze_after_epoch and
                hasattr(model, 'unfreeze_encoder') and
                not encoder_unfrozen):
                logger.info(f"Unfreezing encoder at epoch {epoch}")
                model.unfreeze_encoder()
                encoder_unfrozen = True

                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f} for fine-tuning")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                max_grad_norm=max_grad_norm
            )

            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device, epoch,
                iou_threshold=iou_threshold,
                val_tile_list=val_tiles
            )

            # Track metrics
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

            # Report to Optuna for pruning (Q7: Use both early stopping mechanisms)
            if trial is not None:
                if optuna is None:
                    raise ImportError("Optuna is required for hyperparameter tuning. Install with: poetry install")
                trial.report(val_metrics["val_loss"], epoch)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    if hasattr(trial, "set_user_attr"):
                        try:
                            trial.set_user_attr("pruned_epoch", int(epoch))
                            trial.set_user_attr("pruned_val_loss", float(val_metrics["val_loss"]))
                        except Exception:
                            pass
                    # Print MLflow run info at the moment of pruning (end-of-run for this trial).
                    active = mlflow.active_run()
                    if active is not None:
                        exp_id = active.info.experiment_id
                        run_id = active.info.run_id
                        tracking_uri = mlflow.get_tracking_uri()
                        print(
                            f"[MLFLOW][OPTUNA][PRUNED] trial={trial.number} epoch={epoch} "
                            f"experiment_id={exp_id} run_id={run_id} run_name={run_name} "
                            f"val_loss={val_metrics['val_loss']:.6f}",
                            flush=True,
                        )
                        print(f"[MLFLOW] tracking_uri={tracking_uri}", flush=True)
                    raise optuna.TrialPruned()

            # Update learning rate scheduler
            if lr_scheduler:
                lr_scheduler.step(val_metrics["val_loss"])
                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # Early stopping check
            if early_stopping_patience:
                if val_metrics["val_loss"] < best_val_loss_for_early_stop - early_stopping_min_delta:
                    best_val_loss_for_early_stop = val_metrics["val_loss"]
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"[EARLY STOP] No improvement for {early_stopping_patience} epochs. Stopping training.")
                    print(
                        f"[EARLY STOP] epoch={epoch} "
                        f"val_loss={val_metrics['val_loss']:.6f} "
                        f"best_val_loss={best_val_loss_for_early_stop:.6f} "
                        f"patience={early_stopping_patience}",
                        flush=True,
                    )
                    break

            # Log to MLflow
            all_metrics = {**train_metrics, **val_metrics}
            log_metrics(all_metrics, step=epoch)

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                prev_best = best_val_loss
                best_val_loss = val_metrics["val_loss"]
                best_val_mae = val_metrics["val_mae"]
                best_val_iou = val_metrics.get("val_iou", 0.0)
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, best_model_path
                )
                logger.info(f"New best model saved! val_loss: {best_val_loss:.4f} | "
                          f"val_mae: {best_val_mae:.4f} | val_iou: {best_val_iou:.4f}")
                print(
                    f"[BEST] epoch={epoch} val_loss={best_val_loss:.6f} "
                    f"(prev_best={prev_best:.6f}) val_mae={best_val_mae:.6f} val_iou={best_val_iou:.6f}",
                    flush=True,
                )

            # One-line epoch summary (helps spot early-stop / baseline progress)
            summary = (
                f"[EPOCH] {epoch}/{config['training']['num_epochs']} "
                f"train_loss={train_metrics['train_loss']:.6f} "
                f"val_loss={val_metrics['val_loss']:.6f} "
                f"val_mae={val_metrics['val_mae']:.6f} "
                f"val_iou={val_metrics.get('val_iou', 0.0):.6f} "
                f"best_val_loss={best_val_loss:.6f}"
            )
            if early_stopping_patience:
                summary += f" early_stop={early_stopping_counter}/{early_stopping_patience}"
            if "val_baseline_mae" in val_metrics:
                baseline = float(val_metrics["val_baseline_mae"])
                improvement = float(val_metrics["val_improvement_over_baseline"])
                pct = (improvement / baseline * 100.0) if baseline > 0 else 0.0
                summary += (
                    f" baseline_mae={baseline:.6f} "
                    f"improvement={improvement:+.6f} ({pct:+.1f}%)"
                )
            print(summary, flush=True)

        # Log final metrics to MLflow
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

        # Print MLflow run info at the end of the run (what you can copy/paste).
        active = mlflow.active_run()
        if active is not None:
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

        # Create and log visualization plots (skip for Optuna trials to save time)
        if trial is None:
            logger.info("=== Creating Training Plots ===")
            figures = create_training_plots(metrics_history, baseline_mae)

            for plot_name, fig in figures.items():
                mlflow.log_figure(fig, f"plots/{plot_name}.png")
                plt.close(fig)
                logger.info(f"Logged {plot_name} plot to MLflow")

        # Save model to MLflow (skip for Optuna trials to save time)
        if mlflow_config.get("log_model", True) and trial is None:
            model_size_mb = save_model(model, "model")
            if model_size_mb > 0:
                logger.info(f"Model saved to MLflow (size: {model_size_mb:.2f} MB)")

    return best_val_loss


def main():
    """Main training function."""
    import argparse

    project_root = get_project_root(__file__)

    parser = argparse.ArgumentParser(description="Train CNN model for lobe detection")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev tiles (cropped 1024x1024) instead of full dataset",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: auto-generated)",
    )

    args = parser.parse_args()

    # Load config
    config_path = resolve_path(args.config, project_root)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Determine mode
    mode = "dev" if args.dev else "production"

    # Call the extracted training function
    train_model_with_config(
        config=config,
        mode=mode,
        trial=None,  # Not using Optuna
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
