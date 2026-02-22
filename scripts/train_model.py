#!/usr/bin/env python3
"""
Train CNN model for lobe detection.
"""

import logging
import sys
import time
import warnings
import yaml
from datetime import datetime
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
from src.training.loss_factory import create_criterion
from src.training.dataloader import (
    load_filtered_tiles,
    load_extended_training_tiles,
    create_data_splits,
    create_dataloaders,
)
from src.training.trainer import train_one_epoch, validate, save_checkpoint
from src.training.visualization import (
    create_training_plots,
    plot_loss_simple,
    get_representative_tile_ids_for_viz,
    resolve_representative_tiles,
    create_prediction_tile_figures,
    create_representative_tiles_channel_figures,
    show_best_predicted_tile,
    show_highest_iou_tile,
)
from src.preprocessing.normalization import compute_statistics
from src.utils.mlflow_utils import setup_mlflow_experiment, log_training_config, save_model, log_metrics, get_intention_suggestion
from src.utils.path_utils import get_project_root, resolve_path
from src.utils.config_utils import (
    apply_best_hyperparameters,
    apply_hyperparameters_from_mlflow_run,
    get_training_path_key,
    APPLIED_HP_DISPLAY_KEYS,
)
from src.utils.proximity_utils import infer_proximity_token, detect_proximity_params
from src.utils.voice_notify import notify_training_finished

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
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
    applied_best_hparams: Optional[dict] = None,
    max_tiles: Optional[int] = None,
) -> float:
    """
    Train model with given configuration.

    This function can be called directly or used by Optuna for hyperparameter tuning.

    Args:
        config: Training configuration dictionary
        mode: "dev" or "production"
        trial: Optional Optuna trial object (for pruning)
        run_name: Optional MLflow run name
        applied_best_hparams: Optional dict from best_hyperparameters.yaml (for logging when --best-hparams was used)

    Returns:
        Best validation loss
    """
    project_root = get_project_root(Path(__file__))

    if mode == "synthetic_parenthesis":
        config["data"]["use_background_and_augmentation"] = False

    tile_size = config["data"].get("tile_size", 256)
    target_mode = (config.get("target_mode") or "proximity").lower()
    binary_threshold = float(config.get("binary_threshold", 1.0))
    path_key = get_training_path_key(mode, tile_size)
    paths = config["paths"][path_key]

    filtered_tiles_path = resolve_path(Path(paths["filtered_tiles"]), project_root)
    features_dir = resolve_path(Path(paths["features_dir"]), project_root)
    targets_dir = resolve_path(Path(paths["targets_dir"]), project_root)
    models_dir = resolve_path(Path(paths["models_dir"]), project_root)
    use_segmentation_layer = config["data"].get("use_segmentation_layer", False)
    segmentation_dir = None
    if use_segmentation_layer and paths.get("segmentation_dir"):
        segmentation_dir = resolve_path(Path(paths["segmentation_dir"]), project_root)

    logger.info("=== Training Configuration ===")
    logger.info(f"Mode: {mode} (tile size: {tile_size}x{tile_size}), target_mode: {target_mode}")
    logger.info(f"Filtered tiles: {filtered_tiles_path}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Targets dir: {targets_dir}")
    logger.info(f"Models dir: {models_dir}")
    if use_segmentation_layer:
        if not segmentation_dir or not segmentation_dir.exists():
            raise ValueError(
                "use_segmentation_layer is true but segmentation_dir is missing or does not exist. "
                "Add paths.<key>.segmentation_dir and tile the segmentation raster with the same tile size/overlap."
            )
        logger.info(f"Segmentation dir: {segmentation_dir}")

    # Load filtered tiles
    logger.info("Loading filtered tiles...")
    all_tiles = load_filtered_tiles(filtered_tiles_path)
    if max_tiles is not None:
        viz_config = config.get("visualization", {})
        rep_tile_ids = get_representative_tile_ids_for_viz(viz_config, path_key, tile_size)
        rep_tiles = resolve_representative_tiles(all_tiles, rep_tile_ids) if rep_tile_ids else []
        rep_ids_set = {t.get("tile_id") for t in rep_tiles if t.get("tile_id")}
        remaining = [t for t in all_tiles if t.get("tile_id") not in rep_ids_set]
        n_rep = len(rep_tiles)
        n_remaining = max(0, max_tiles - n_rep)
        rng = np.random.default_rng(42)
        if n_remaining > 0 and remaining:
            n_take = min(n_remaining, len(remaining))
            indices = rng.choice(len(remaining), size=n_take, replace=False)
            sampled = [remaining[i] for i in indices]
            all_tiles = rep_tiles + sampled
        else:
            all_tiles = rep_tiles if rep_tiles else all_tiles[:max_tiles]
        logger.info(
            f"Capped to {max_tiles} tiles (included {n_rep} representative, "
            f"+ {len(all_tiles) - n_rep} random; total: {len(all_tiles)})"
        )
    else:
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

    use_bg_aug = config["data"].get("use_background_and_augmentation", False)
    if use_bg_aug:
        extended_path = filtered_tiles_path.parent / "extended_training_tiles.json"
        if extended_path.exists():
            train_tiles, ext_config, ext_stats = load_extended_training_tiles(extended_path)
            logger.info(
                f"Loaded extended training set from {extended_path}: {len(train_tiles)} tiles "
                f"(stats: {ext_stats})"
            )
        else:
            logger.warning(
                "use_background_and_augmentation is true but extended_training_tiles.json not found at %s; "
                "using filtered_tiles split only. For background + augmentation run prepare_extended_training_set.py.",
                extended_path,
            )

    # Compute normalization statistics from lobe training tiles only (exclude background/augmented for stats)
    logger.info("Computing normalization statistics...")
    extended_loaded = use_bg_aug and (filtered_tiles_path.parent / "extended_training_tiles.json").exists()
    tiles_for_stats = [t for t in train_tiles if t.get("role") == "lobe"] if extended_loaded else train_tiles
    train_feature_paths = [features_dir / tile["features_path"] for tile in tiles_for_stats]
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

    train_subsample_ratio = config["data"].get("train_subsample_ratio", 1.0)
    if train_subsample_ratio < 1.0:
        logger.info(
            "Train subsampling: %.0f%% of tiles per epoch (new random subset each epoch)",
            100 * train_subsample_ratio,
        )

    # Create data loaders (no on-the-fly augmentation when using extended set; augmentation is pre-written)
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_tiles,
        val_tiles,
        features_dir,
        targets_dir,
        normalization_stats,
        batch_size=config["training"]["batch_size"],
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_dir,
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model using factory (binary mode: output [0,1] via sigmoid)
    logger.info("Creating model...")
    model_config = dict(config["model"])
    if use_segmentation_layer:
        model_config["in_channels"] = 6
    if target_mode == "binary":
        model_config["proximity_max"] = 1
        model_config["output_activation"] = "sigmoid"
    model = create_model(model_config).to(device)

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

    iou_threshold = (
        0.5 if target_mode == "binary"
        else config["training"].get("iou_threshold", 5.0)
    )
    criterion = create_criterion(config["training"], target_mode=target_mode)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Setup learning rate scheduler (use null or "none" for fixed LR)
    lr_scheduler = None
    lr_scheduler_config = config["training"].get("lr_scheduler")
    if lr_scheduler_config and str(lr_scheduler_config).lower() != "none":
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
    else:
        logger.info("No LR scheduler (fixed learning rate)")

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
        loss_plot_path = None  # set when file: tracking so we can write loss.png each epoch

        # Print run location at start so you can follow plots during training
        active = mlflow.active_run()
        if active is not None:
            exp_id = active.info.experiment_id
            run_id = active.info.run_id
            tracking_uri = mlflow.get_tracking_uri()
            print(
                f"[MLFLOW][START] experiment_id={exp_id} run_id={run_id} run_name={run_name}",
                flush=True,
            )
            if tracking_uri.startswith("file:"):
                base = tracking_uri.replace("file:", "").rstrip("/")
                plots_dir = Path(base) / str(exp_id) / run_id / "artifacts" / "plots"
                loss_plot_path = plots_dir / "loss.png"
                plots_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Plots (updated each epoch): {loss_plot_path}", flush=True)
            print(f"  tracking_uri={tracking_uri}", flush=True)

        # Log config
        log_training_config(config)
        mlflow.log_param("mode", mode)
        mlflow.log_param("target_mode", target_mode)
        mlflow.log_param("num_params", num_params)
        if applied_best_hparams is not None:
            hp = applied_best_hparams.get("hyperparameters") or {}
            if applied_best_hparams.get("source") == "mlflow_run":
                run_id = applied_best_hparams.get("run_id")
                best_val = applied_best_hparams.get("best_validation_loss")
                mlflow.set_tag("hp_source_run_id", str(run_id))
                if best_val is not None:
                    mlflow.log_param("hp_from_run.best_validation_loss", best_val)
                for k, v in hp.items():
                    mlflow.log_param(f"hp_from_run.{k}", v)
                logger.info(
                    "Using hyperparameters from MLflow run_id=%s val_loss=%s loss=%s lr=%s batch=%s encoder=%s unfreeze_epoch=%s",
                    run_id, best_val, hp.get("loss_function"), hp.get("learning_rate"), hp.get("batch_size"),
                    hp.get("encoder_name"), hp.get("unfreeze_after_epoch"),
                )
            else:
                trial_num = applied_best_hparams.get("best_trial_number")
                best_val = applied_best_hparams.get("best_validation_loss")
                mlflow.set_tag("best_hparams_trial", str(trial_num))
                mlflow.log_param("best_hparams.best_validation_loss", best_val)
                mlflow.log_param("best_hparams.best_trial_number", trial_num)
                for k, v in hp.items():
                    mlflow.log_param(f"best_hparams.{k}", v)
                logger.info(
                    "Using best hyperparameters: trial=%s val_loss=%.4f loss=%s lr=%.2e batch=%s encoder=%s unfreeze_epoch=%s",
                    trial_num, best_val,
                    hp.get("loss_function"), hp.get("learning_rate"), hp.get("batch_size"),
                    hp.get("encoder_name"), hp.get("unfreeze_after_epoch"),
                )
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("num_train_tiles", len(train_tiles))
        mlflow.log_param("num_val_tiles", len(val_tiles))
        if train_subsample_ratio < 1.0:
            mlflow.log_param("data.train_subsample_ratio", train_subsample_ratio)

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
                trial.set_user_attr("model_in_channels", 6 if use_segmentation_layer else int(config["model"].get("in_channels", 5)))
                trial.set_user_attr("model_out_channels", int(config["model"].get("out_channels", 1)))
                trial.set_user_attr("training_iou_threshold", float(iou_threshold))
                trial.set_user_attr("data_normalize_rgb", bool(config["data"].get("normalize_rgb", True)))
                trial.set_user_attr("data_standardize_dem", bool(config["data"].get("standardize_dem", True)))
                trial.set_user_attr("data_standardize_slope", bool(config["data"].get("standardize_slope", True)))
                trial.set_user_attr("filtered_tiles_path", str(filtered_tiles_path))
                trial.set_user_attr("features_dir", str(features_dir))
                trial.set_user_attr("targets_dir", str(targets_dir))
                trial.set_user_attr("proximity_token", infer_proximity_token(str(targets_dir)))
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

        proximity_max_value, proximity_max_distance = detect_proximity_params(
            targets_dir, val_tiles, sample_size=5
        )
        if proximity_max_value is not None:
            mlflow.log_param("data.proximity_max_value", proximity_max_value)
            mlflow.log_param("data.proximity_max_distance", proximity_max_distance)
            logger.info(f"Detected proximity map: max_value={proximity_max_value}, max_distance={proximity_max_distance}")

        # Training loop
        best_val_loss = float("inf")
        best_model_path = models_dir / "best_model.pt"
        iou_threshold = config["training"].get("iou_threshold", 5.0)
        best_tile_loss_so_far = float("inf")
        best_tile_info_so_far = None
        best_iou_so_far = -1.0
        best_iou_tile_info_so_far = None
        best_iou_tile_loss_so_far = None
        last_epoch_saved = None
        last_drawn_best_tile_id = None
        last_drawn_best_tile_loss = None
        last_drawn_best_iou_tile_id = None
        last_drawn_best_iou = None
        SAVE_THROTTLE_EPOCHS = 10

        # Track metrics across epochs for visualization
        metrics_history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_iou": [],
            "improvement_percent": [],
            "early_stop_counter": [],
        }
        baseline_mae = None
        enc = config.get("model", {}).get("encoder", {})
        tr = config.get("training", {})
        da = config.get("data", {})

        def _fmt(v):
            if v is None:
                return "?"
            if isinstance(v, float):
                s = f"{v:.10f}".rstrip("0").rstrip(".")
                return s if s else "0"
            return str(v)

        out_act = "sigmoid" if target_mode == "binary" else config.get("model", {}).get("output_activation", "?")
        loss_plot_config_parts = [
            f"arch={config.get('model', {}).get('architecture', '?')}",
            f"drop={_fmt(config.get('model', {}).get('dropout'))}",
            f"enc={enc.get('name', '?')}",
            f"pretrain={enc.get('pretrained', '?')}",
            f"freeze={enc.get('freeze_encoder', '?')}",
            f"unfreeze_ep={enc.get('unfreeze_after_epoch', '?')}",
            f"dec_drop={_fmt(config.get('model', {}).get('decoder_dropout'))}",
            f"out_act={out_act}",
            f"batch={tr.get('batch_size', '?')}",
            f"ep={tr.get('num_epochs', '?')}",
            f"lr={_fmt(tr.get('learning_rate'))}",
            f"loss={tr.get('loss_function', '?')}",
            f"tile={da.get('tile_size', '?')}",
            f"train_split={_fmt(da.get('train_split'))}",
        ]
        loss_plot_options = {
            "config_summary": " | ".join(loss_plot_config_parts),
            "num_train_tiles": len(train_tiles),
            "num_val_tiles": len(val_tiles),
            "early_stopping_patience": early_stopping_patience,
            "freeze_encoder": enc.get("freeze_encoder"),
            "unfreeze_after_epoch": enc.get("unfreeze_after_epoch"),
        }

        logger.info("=== Starting Training ===")
        logger.info(f"IoU threshold: {iou_threshold}")

        if trial is None:
            run_intention = None
            active = mlflow.active_run()
            if active is not None:
                suggestion = get_intention_suggestion(active.info.experiment_id, active.info.run_id)
                if suggestion:
                    print(f"Suggestion: {suggestion}", flush=True)
            try:
                run_intention = input("Run intention (subtitle): ").strip() or None
            except EOFError:
                run_intention = None
            loss_plot_options["run_intention"] = run_intention
        else:
            loss_plot_options["run_intention"] = None

        # Create initial placeholder so the printed path exists before first epoch (training and Optuna trials)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training started – plot will update after each epoch.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        mlflow.log_figure(fig, "plots/loss.png")
        plt.close(fig)

        # Encoder unfreezing configuration
        encoder_config = config["model"].get("encoder", {})
        unfreeze_after_epoch = encoder_config.get("unfreeze_after_epoch", 0)
        encoder_unfrozen = False

        training_start_time = time.time()
        loss_plot_options["training_start_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        for epoch in range(1, config["training"]["num_epochs"] + 1):
            if train_subsample_ratio < 1.0:
                rng = np.random.default_rng(epoch)
                n = max(1, int(len(train_tiles) * train_subsample_ratio))
                indices = rng.choice(len(train_tiles), size=n, replace=False)
                epoch_tiles = [train_tiles[i] for i in indices]
                train_loader, _ = create_dataloaders(
                    epoch_tiles,
                    val_tiles,
                    features_dir,
                    targets_dir,
                    normalization_stats,
                    batch_size=config["training"]["batch_size"],
                    tile_size=tile_size,
                    target_mode=target_mode,
                    binary_threshold=binary_threshold,
                    segmentation_base_dir=segmentation_dir,
                )

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
            val_metrics, best_tile_result, best_iou_tile_result, _ = validate(
                model, val_loader, criterion, device, epoch,
                iou_threshold=iou_threshold,
                val_tile_list=val_tiles,
                return_best_tile=(trial is None),
                return_batch_losses=False,
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
                metrics_history["early_stop_counter"].append(early_stopping_counter)

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

            # Track best tile by loss and by IoU (before save so we can draw with up-to-date best when we do save)
            if trial is None and best_tile_result is not None:
                tile_info, tile_loss = best_tile_result
                if tile_loss < best_tile_loss_so_far:
                    best_tile_loss_so_far = tile_loss
                    best_tile_info_so_far = tile_info
                    logger.info(
                        "New lowest-loss tile: %s loss=%.6f",
                        best_tile_info_so_far.get("tile_id", "?"),
                        best_tile_loss_so_far,
                    )
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
                    logger.info(
                        "New highest IoU tile: %s IoU=%.4f",
                        best_iou_tile_info_so_far.get("tile_id", "?"),
                        best_iou_so_far,
                    )

            # Save best model to disk and redraw best-tile figures only when new best val_loss AND 10+ epochs since last save
            if val_metrics["val_loss"] < best_val_loss:
                prev_best = best_val_loss
                best_val_loss = val_metrics["val_loss"]
                best_val_mae = val_metrics["val_mae"]
                best_val_iou = val_metrics.get("val_iou", 0.0)
                may_save = (
                    last_epoch_saved is None
                    or (epoch - last_epoch_saved) >= SAVE_THROTTLE_EPOCHS
                )
                if may_save:
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics, best_model_path
                    )
                    logger.info(f"New best model saved! val_loss: {best_val_loss:.4f} | "
                              f"val_mae: {best_val_mae:.4f} | val_iou: {best_val_iou:.4f}")
                    if trial is None and best_tile_info_so_far is not None:
                        bid = best_tile_info_so_far.get("tile_id")
                        if bid != last_drawn_best_tile_id or best_tile_loss_so_far != last_drawn_best_tile_loss:
                            logger.info("Creating best predicted tile figure")
                            fig = show_best_predicted_tile(
                                model,
                                best_tile_info_so_far,
                                features_dir,
                                targets_dir,
                                normalization_stats,
                                device,
                                tile_size,
                                iou_threshold,
                                best_tile_loss_so_far,
                                target_mode=target_mode,
                                binary_threshold=binary_threshold,
                                segmentation_base_dir=segmentation_dir,
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
                            logger.info("Creating best IoU tile figure")
                            fig = show_highest_iou_tile(
                                model,
                                best_iou_tile_info_so_far,
                                features_dir,
                                targets_dir,
                                normalization_stats,
                                device,
                                tile_size,
                                iou_threshold,
                                best_iou_so_far,
                                target_mode=target_mode,
                                binary_threshold=binary_threshold,
                                segmentation_base_dir=segmentation_dir,
                                tile_loss=best_iou_tile_loss_so_far,
                            )
                            mlflow.log_figure(fig, "plots/best_iou_tile.png")
                            if loss_plot_path is not None:
                                fig.savefig(loss_plot_path.parent / "best_iou_tile.png", dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            last_drawn_best_iou_tile_id = iid
                            last_drawn_best_iou = best_iou_so_far
                    last_epoch_saved = epoch
                print(
                    f"[BEST] epoch={epoch} val_loss={best_val_loss:.6f} "
                    f"(prev_best={prev_best:.6f}) val_mae={best_val_mae:.6f} val_iou={best_val_iou:.6f}"
                    + (" (saved)" if may_save else " (not saved, throttle)"),
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

            # Update simple loss plot each epoch (fast); advanced plot is created at end of run
            if not early_stopping_patience:
                metrics_history["early_stop_counter"].append(0)
            loss_fig = plot_loss_simple(
                metrics_history["epochs"],
                metrics_history["train_loss"],
                metrics_history["val_loss"],
            )
            mlflow.log_figure(loss_fig, "plots/loss.png")
            if loss_plot_path is not None:
                loss_fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
            plt.close(loss_fig)

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

        if trial is None:
            elapsed_seconds = time.time() - training_start_time
            notify_training_finished(elapsed_seconds, epoch)

        # For non-Optuna runs, load best checkpoint so plots, prediction viz, and MLflow model use it
        if trial is None and best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best checkpoint for plots and MLflow")

        # Create and log full visualization plots including advanced loss; overwrite loss.png with advanced version
        if trial is None:
            logger.info("=== Creating Training Plots ===")
            loss_plot_options["training_duration_seconds"] = elapsed_seconds
            figures = create_training_plots(metrics_history, baseline_mae, loss_plot_options=loss_plot_options)

            for plot_name, fig in figures.items():
                mlflow.log_figure(fig, f"plots/{plot_name}.png")
                if plot_name == "loss" and loss_plot_path is not None:
                    fig.savefig(loss_plot_path, dpi=150)
                plt.close(fig)
                logger.info(f"Logged {plot_name} plot to MLflow")

            # Best-tile figures (deferred from per-epoch to avoid slow "new best" epochs)
            if best_tile_info_so_far is not None:
                logger.info("=== Creating best predicted tile figure ===")
                fig = show_best_predicted_tile(
                    model,
                    best_tile_info_so_far,
                    features_dir,
                    targets_dir,
                    normalization_stats,
                    device,
                    tile_size,
                    iou_threshold,
                    best_tile_loss_so_far,
                    target_mode=target_mode,
                    binary_threshold=binary_threshold,
                    segmentation_base_dir=segmentation_dir,
                )
                mlflow.log_figure(fig, "plots/best_predicted_tile.png")
                if loss_plot_path is not None:
                    fig.savefig(loss_plot_path.parent / "best_predicted_tile.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            if best_iou_tile_info_so_far is not None:
                logger.info("=== Creating best IoU tile figure ===")
                fig = show_highest_iou_tile(
                    model,
                    best_iou_tile_info_so_far,
                    features_dir,
                    targets_dir,
                    normalization_stats,
                    device,
                    tile_size,
                    iou_threshold,
                    best_iou_so_far,
                    target_mode=target_mode,
                    binary_threshold=binary_threshold,
                    segmentation_base_dir=segmentation_dir,
                    tile_loss=best_iou_tile_loss_so_far,
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
                )
                for tid, fig in pred_figures.items():
                    mlflow.log_figure(fig, f"prediction_tiles/{tid}.png")
                    plt.close(fig)
                    logger.info(f"Logged prediction tile: {tid}")
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
                    plot_options=loss_plot_options,
                )
                for tid, fig in channel_figures.items():
                    mlflow.log_figure(fig, f"representative_channels/{tid}.png")
                    plt.close(fig)
                    logger.info(f"Logged representative channels: {tid}")
            elif rep_tile_ids:
                logger.warning(
                    "representative_tile_ids configured but no matching tiles found and fallback_n=0; "
                    "check tile IDs match filtered_tiles.json or set prediction_tiles_fallback_n > 0"
                )

        # Save model to MLflow (skip for Optuna trials to save time)
        if mlflow_config.get("log_model", True) and trial is None:
            model_size_mb = save_model(model, "model")
            if model_size_mb > 0:
                logger.info(f"Model saved to MLflow (size: {model_size_mb:.2f} MB)")

    return best_val_loss


def _print_applied_hyperparameters(applied_best_hparams: dict, header: str) -> None:
    if not applied_best_hparams:
        return
    hp = applied_best_hparams.get("hyperparameters") or {}
    run_id = applied_best_hparams.get("run_id")
    trial_num = applied_best_hparams.get("best_trial_number")
    best_val = applied_best_hparams.get("best_validation_loss")
    print("")
    print(header)
    if run_id is not None:
        print(f"  run_id: {run_id}")
    if trial_num is not None:
        print(f"  best_trial_number: {trial_num}")
    print(f"  best_validation_loss: {best_val}")
    for key in APPLIED_HP_DISPLAY_KEYS:
        if key in hp:
            print(f"  {key}: {hp[key]}")
    print("=" * (len(header) if len(header) > 40 else 40))
    print("")


def main():
    """Main training function."""
    import argparse

    project_root = get_project_root(Path(__file__))

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
        "--mode",
        type=str,
        choices=["dev", "production", "synthetic_parenthesis"],
        default=None,
        help="Dataset mode: dev, production, or synthetic_parenthesis (sanity-check). Default: dev if --dev else production.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: auto-generated)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override num_epochs from config (e.g. 1 for a quick dry run)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        choices=[256, 512],
        help="Override data tile size from config (e.g. 256 for dev when dev/train_512 does not exist)",
    )
    parser.add_argument(
        "--best-hparams",
        action="store_true",
        help="Override config with best hyperparameters from configs/best_hyperparameters.yaml",
    )
    parser.add_argument(
        "--best-hparams-path",
        type=Path,
        default=None,
        help="Path to best hyperparameters YAML (default: configs/best_hyperparameters.yaml)",
    )
    parser.add_argument(
        "--hp_from_run_id",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Apply hyperparameters from an MLflow run ID (e.g. from MLflow UI).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        metavar="N",
        help="Max tiles to use in total (before train/val/test split). Use for quick runs.",
    )

    args = parser.parse_args()

    # Load config
    config_path = resolve_path(args.config, project_root)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.max_epochs is not None:
        config["training"]["num_epochs"] = args.max_epochs
    if args.tile_size is not None:
        config["data"]["tile_size"] = args.tile_size
    applied_best_hparams = None
    tracking_uri = config.get("mlflow", {}).get("tracking_uri")
    if args.hp_from_run_id:
        applied_best_hparams = apply_hyperparameters_from_mlflow_run(
            config, args.hp_from_run_id, tracking_uri=tracking_uri
        )
        _print_applied_hyperparameters(
            applied_best_hparams,
            "=== Applied hyperparameters from MLflow run ===",
        )
    elif args.best_hparams:
        best_hp_path = args.best_hparams_path or (project_root / "configs" / "best_hyperparameters.yaml")
        best_hp_path = resolve_path(best_hp_path, project_root)
        if not best_hp_path.exists():
            raise FileNotFoundError(
                f"Best hyperparameters file not found: {best_hp_path}. "
                "Run tune_hyperparameters.py first or set --best-hparams-path."
            )
        applied_best_hparams = apply_best_hyperparameters(config, best_hp_path)
        _print_applied_hyperparameters(
            applied_best_hparams,
            "=== Applied best hyperparameters (from best_hyperparameters.yaml) ===",
        )

    # Determine mode
    if args.mode is not None:
        mode = args.mode
    else:
        mode = "dev" if args.dev else "production"

    # Call the extracted training function
    train_model_with_config(
        config=config,
        mode=mode,
        trial=None,  # Not using Optuna
        run_name=args.run_name,
        applied_best_hparams=applied_best_hparams,
        max_tiles=args.max_tiles,
    )


if __name__ == "__main__":
    main()
