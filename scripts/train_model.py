#!/usr/bin/env python3
"""
Train CNN model for lobe detection.
"""

import copy
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

import torch
import mlflow

from src.training.dataloader import create_dataloaders
from src.utils.mlflow_utils import setup_mlflow_experiment
from src.utils.path_utils import get_project_root, resolve_path
from src.utils.config_utils import (
    apply_best_hyperparameters,
    apply_hyperparameters_from_mlflow_run,
    APPLIED_HP_DISPLAY_KEYS,
)
from src.training.training_config import (
    resolve_training_paths,
    compute_in_channels,
    get_normalization_stats,
    log_training_config_summary,
)
from src.training.setup import (
    prepare_tiles_and_splits,
    create_training_dataloaders,
    build_model_and_training_components,
)
from src.training.mlflow_run_context import (
    get_loss_plot_path_and_print_run_start,
    log_run_config_and_trial_metadata,
    create_initial_loss_placeholder,
    build_loss_plot_options,
    prompt_run_intention,
)
from src.training.cli import build_train_parser, apply_cli_overrides
from src.training.trainer import load_training_checkpoint
from src.training.training_loop import run_training_loop
from src.training.post_training import (
    log_final_metrics_and_trial_attrs,
    print_mlflow_run_end,
    load_best_checkpoint,
    run_post_training_visualization,
    save_mlflow_model_if_enabled,
)
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
    filtered_tiles_override: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    config_path: Optional[Path] = None,
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
    resolved = resolve_training_paths(config, mode, project_root, filtered_tiles_override)
    in_channels = compute_in_channels(config["data"])
    log_training_config_summary(resolved, mode)

    filtered_tiles_path = resolved.filtered_tiles_path
    features_dir = resolved.features_dir
    targets_dir = resolved.targets_dir
    models_dir = resolved.models_dir
    segmentation_dir = resolved.segmentation_dir
    slope_stripes_channel_dir = resolved.slope_stripes_channel_dir
    path_key = resolved.path_key
    tile_size = resolved.tile_size
    target_mode = resolved.target_mode
    binary_threshold = resolved.binary_threshold
    use_rgb = resolved.use_rgb
    use_dem = resolved.use_dem
    use_slope = resolved.use_slope

    train_tiles, val_tiles, test_tiles, all_tiles = prepare_tiles_and_splits(
        config, resolved, max_tiles
    )
    extended_path = resolved.filtered_tiles_path.parent / "extended_training_tiles.json"
    use_bg_aug = config["data"].get("use_background_and_augmentation", False)
    normalization_stats = get_normalization_stats(
        train_tiles, resolved.features_dir,
        use_rgb, use_dem, use_slope,
        use_bg_aug, extended_path,
    )
    if not normalization_stats:
        logger.info("Skipping normalization statistics (no RGB/DEM/Slope channels).")

    train_subsample_ratio = config["data"].get("train_subsample_ratio", 1.0)
    if train_subsample_ratio < 1.0:
        logger.info(
            "Train subsampling: %.0f%% of tiles per epoch (new random subset each epoch)",
            100 * train_subsample_ratio,
        )

    logger.info("Creating data loaders...")
    train_loader, val_loader = create_training_dataloaders(
        train_tiles, val_tiles, resolved, normalization_stats, config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    components = build_model_and_training_components(
        config, in_channels, target_mode, device
    )
    model = components.model
    criterion = components.criterion
    optimizer = components.optimizer
    lr_scheduler = components.lr_scheduler

    resume_loop_state: Optional[dict] = None
    if resume_from is not None:
        if trial is not None:
            raise ValueError("Resume is not supported together with Optuna trial.")
        ckpt_path = resolve_path(resume_from, project_root)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = load_training_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if lr_scheduler is not None and ckpt.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        resume_loop_state = ckpt.get("training_loop_state")
        if resume_loop_state is None:
            raise ValueError(
                "Checkpoint has no training_loop_state. Use training_latest.pt or a best_model.pt "
                "saved by a current training run (full resume format)."
            )
        logger.info(
            "Loaded resume checkpoint from %s (last completed epoch %s)",
            ckpt_path,
            resume_loop_state.get("last_completed_epoch"),
        )
    iou_threshold = components.iou_threshold
    early_stopping_patience = components.early_stopping_patience
    early_stopping_min_delta = components.early_stopping_min_delta
    max_grad_norm = components.max_grad_norm
    num_params = components.num_params
    trainable_params = components.trainable_params
    architecture = components.architecture

    # Setup MLflow
    mlflow_config = config["mlflow"]
    setup_mlflow_experiment(mlflow_config["experiment_name"], mlflow_config.get("tracking_uri"))

    if run_name is None:
        run_name = f"unet_baseline_{mode}"

    with mlflow.start_run(run_name=run_name):
        loss_plot_path = get_loss_plot_path_and_print_run_start(run_name)
        log_run_config_and_trial_metadata(
            config, mode, run_name, trial, applied_best_hparams,
            num_params, trainable_params, len(train_tiles), len(val_tiles), train_subsample_ratio,
            architecture, in_channels, iou_threshold,
            filtered_tiles_path, features_dir, targets_dir, val_tiles,
        )

        best_model_path = models_dir / "best_model.pt"
        loss_plot_options = build_loss_plot_options(
            config, len(train_tiles), len(val_tiles),
            early_stopping_patience, early_stopping_min_delta,
        )
        loss_plot_options["run_intention"] = prompt_run_intention(trial)

        logger.info("=== Starting Training ===")
        logger.info("IoU threshold: %s", iou_threshold)
        create_initial_loss_placeholder()

        def _make_loaders(tr, val):
            return create_dataloaders(
                tr, val, features_dir, targets_dir, normalization_stats,
                batch_size=config["training"]["batch_size"],
                tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
                segmentation_base_dir=segmentation_dir, slope_stripes_base_dir=slope_stripes_channel_dir,
                use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
            )

        training_start_time = time.time()
        loss_plot_options["training_start_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if resume_loop_state is not None:
            mlflow.set_tag("resumed_from_checkpoint", str(resume_from))
            mlflow.log_param(
                "resume_last_completed_epoch",
                int(resume_loop_state.get("last_completed_epoch", 0)),
            )
        config_snapshot = copy.deepcopy(config)
        result = run_training_loop(
            config, model, criterion, optimizer, lr_scheduler, device,
            train_tiles, val_tiles, train_loader, val_loader, _make_loaders,
            features_dir, targets_dir, normalization_stats, tile_size, target_mode, binary_threshold,
            segmentation_dir, slope_stripes_channel_dir, use_rgb, use_dem, use_slope,
            iou_threshold, early_stopping_patience, early_stopping_min_delta, max_grad_norm,
            best_model_path, loss_plot_path, trial, run_name, train_subsample_ratio, optuna,
            resume_state=resume_loop_state,
            models_dir=models_dir,
            project_root=project_root,
            config_path=config_path,
            mode=mode,
            config_snapshot=config_snapshot,
        )
        best_val_loss = result.best_val_loss
        best_val_mae = result.best_val_mae
        best_val_iou = result.best_val_iou
        epoch = result.last_epoch

        log_final_metrics_and_trial_attrs(best_val_loss, best_val_mae, best_val_iou, trial)
        print_mlflow_run_end(run_name, best_val_loss, trial)

        if trial is None:
            elapsed_seconds = time.time() - training_start_time
            notify_training_finished(elapsed_seconds, epoch)

        load_best_checkpoint(model, best_model_path, device)
        if trial is None:
            run_post_training_visualization(
                config, model, result,
                features_dir, targets_dir, normalization_stats, device,
                tile_size, iou_threshold, target_mode, binary_threshold,
                segmentation_dir, slope_stripes_channel_dir,
                use_rgb, use_dem, use_slope,
                all_tiles, path_key, loss_plot_path, loss_plot_options, elapsed_seconds,
            )
        save_mlflow_model_if_enabled(model, mlflow_config, trial)

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
    project_root = get_project_root(Path(__file__))
    parser = build_train_parser(project_root)
    args = parser.parse_args()

    # Load config
    config_path = resolve_path(args.config, project_root)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    apply_cli_overrides(config, args)

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

    resume_from = getattr(args, "resume", None)
    resume_path = resolve_path(resume_from, project_root) if resume_from else None

    train_model_with_config(
        config=config,
        mode=mode,
        trial=None,  # Not using Optuna
        run_name=args.run_name,
        applied_best_hparams=applied_best_hparams,
        max_tiles=args.max_tiles,
        filtered_tiles_override=args.filtered_tiles,
        resume_from=resume_path,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
