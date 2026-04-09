#!/usr/bin/env python3
"""Train CNN model for lobe detection."""

import copy
import logging
import time
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    validate_in_channels,
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
from src.training.visualization import (
    get_representative_tile_ids_for_viz,
    resolve_representative_tiles,
)
from src.utils.voice_notify import notify_training_finished

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

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
    run_intention: Optional[str] = None,
    init_weights_from: Optional[Path] = None,
) -> float:
    project_root = get_project_root(Path(__file__))
    resolved = resolve_training_paths(config, mode, project_root, filtered_tiles_override)
    in_channels = validate_in_channels(config)
    log_training_config_summary(resolved, mode)

    layer_registry = resolved.layer_registry
    targets_dir = resolved.targets_dir
    models_dir = resolved.models_dir
    tile_size = resolved.tile_size
    target_mode = resolved.target_mode
    binary_threshold = resolved.binary_threshold
    path_key = resolved.path_key

    train_tiles, val_tiles, test_tiles, all_tiles = prepare_tiles_and_splits(
        config, resolved, max_tiles
    )
    extended_path = resolved.filtered_tiles_path.parent / "extended_training_tiles.json"
    use_bg_aug = config["data"].get("use_background_and_augmentation", False)
    normalization_stats = get_normalization_stats(
        train_tiles, layer_registry, use_bg_aug, extended_path,
    )

    train_subsample_ratio = config["data"].get("train_subsample_ratio", 1.0)
    if train_subsample_ratio < 1.0:
        logger.info(
            "Train subsampling: %.0f%% of tiles per epoch (new random subset each epoch)",
            100 * train_subsample_ratio,
        )

    logger.info("Creating data loaders...")
    train_loader, val_loader = create_training_dataloaders(
        train_tiles, val_tiles, resolved, config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {device}")
    components = build_model_and_training_components(
        config, in_channels, target_mode, device
    )
    model = components.model
    criterion = components.criterion
    optimizer = components.optimizer
    lr_scheduler = components.lr_scheduler

    if resume_from is not None and init_weights_from is not None:
        raise ValueError("Cannot use both --resume and --init-weights. Pick one.")

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

    if init_weights_from is not None:
        weights_path = resolve_path(init_weights_from, project_root)
        if not weights_path.exists():
            raise FileNotFoundError(f"Init-weights checkpoint not found: {weights_path}")
        ckpt = load_training_checkpoint(weights_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_epoch = ckpt.get("epoch", "?")
        prev_loss = ckpt.get("metrics", {}).get("val_loss", "?")
        logger.info(
            "Loaded model weights from %s (epoch %s, val_loss %s). "
            "Optimizer, scheduler, and early stopping are fresh.",
            weights_path, prev_epoch, prev_loss,
        )
    iou_threshold = components.iou_threshold
    early_stopping_patience = components.early_stopping_patience
    early_stopping_min_delta = components.early_stopping_min_delta
    max_grad_norm = components.max_grad_norm
    num_params = components.num_params
    trainable_params = components.trainable_params
    architecture = components.architecture

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
            resolved.filtered_tiles_path, targets_dir, val_tiles,
        )

        best_model_path = models_dir / "best_model.pt"
        loss_plot_options = build_loss_plot_options(
            config, len(train_tiles), len(val_tiles),
            early_stopping_patience, early_stopping_min_delta,
        )
        if run_intention is not None:
            loss_plot_options["run_intention"] = run_intention
            active = mlflow.active_run()
            if active is not None:
                mlflow.set_tag("run_intention", run_intention)
        else:
            loss_plot_options["run_intention"] = prompt_run_intention(trial)

        logger.info("=== Starting Training ===")
        logger.info("IoU threshold: %s", iou_threshold)
        create_initial_loss_placeholder()

        _train_aug = config["data"].get("augmentation", False)
        _aug_cfg = config["data"].get("augmentation_config", {})
        _num_workers = int(config["data"].get("dataloader_num_workers", 0))

        def _make_loaders(tr, val):
            return create_dataloaders(
                tr, val, targets_dir, layer_registry,
                batch_size=config["training"]["batch_size"],
                num_workers=_num_workers,
                tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
                train_augmentation=_train_aug, augmentation_config=_aug_cfg,
            )

        training_start_time = time.time()
        loss_plot_options["training_start_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if resume_loop_state is not None:
            mlflow.set_tag("resumed_from_checkpoint", str(resume_from))
            mlflow.log_param(
                "resume_last_completed_epoch",
                int(resume_loop_state.get("last_completed_epoch", 0)),
            )
        if init_weights_from is not None:
            mlflow.set_tag("init_weights_from", str(init_weights_from))
        config_snapshot = copy.deepcopy(config)
        viz_config = config.get("visualization", {})
        rep_tile_ids = get_representative_tile_ids_for_viz(viz_config, path_key, tile_size)
        rep_tiles = resolve_representative_tiles(all_tiles, rep_tile_ids) if rep_tile_ids else []
        fallback_n = int(viz_config.get("prediction_tiles_fallback_n", 0))
        if not rep_tiles and fallback_n > 0 and all_tiles:
            rep_tiles = all_tiles[:fallback_n]
        viz_interval = float(viz_config.get("viz_interval_seconds", 3600))

        result = run_training_loop(
            config, model, criterion, optimizer, lr_scheduler, device,
            train_tiles, val_tiles, train_loader, val_loader, _make_loaders,
            targets_dir, layer_registry,
            tile_size, target_mode, binary_threshold,
            iou_threshold, early_stopping_patience, early_stopping_min_delta, max_grad_norm,
            best_model_path, loss_plot_path, trial, run_name, train_subsample_ratio, optuna,
            resume_state=resume_loop_state,
            models_dir=models_dir,
            project_root=project_root,
            config_path=config_path,
            mode=mode,
            config_snapshot=config_snapshot,
            representative_tiles=rep_tiles,
            viz_interval_seconds=viz_interval,
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
                targets_dir, layer_registry,
                device, tile_size, iou_threshold, target_mode, binary_threshold,
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
    project_root = get_project_root(Path(__file__))
    parser = build_train_parser(project_root)
    args = parser.parse_args()

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

    if args.mode is not None:
        mode = args.mode
    else:
        mode = "dev" if args.dev else "production"

    resume_from = getattr(args, "resume", None)
    resume_path = resolve_path(resume_from, project_root) if resume_from else None
    init_weights = getattr(args, "init_weights", None)
    init_weights_path = resolve_path(init_weights, project_root) if init_weights else None

    train_model_with_config(
        config=config,
        mode=mode,
        trial=None,
        run_name=args.run_name,
        applied_best_hparams=applied_best_hparams,
        max_tiles=args.max_tiles,
        filtered_tiles_override=args.filtered_tiles,
        resume_from=resume_path,
        config_path=config_path,
        init_weights_from=init_weights_path,
    )


if __name__ == "__main__":
    main()
