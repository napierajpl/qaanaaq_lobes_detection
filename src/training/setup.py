"""
Training setup: tile loading/splits, dataloaders, model and training components.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from src.models.factory import create_model
from src.training.dataloader import (
    load_filtered_tiles,
    load_extended_training_tiles,
    create_data_splits,
    create_dataloaders,
)
from src.training.loss_factory import create_criterion
from src.training.training_config import (
    ResolvedTrainingPaths,
    validate_data_splits,
    apply_illumination_filter,
)
from src.training.visualization import (
    get_representative_tile_ids_for_viz,
    resolve_representative_tiles,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingComponents:
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[Any]
    iou_threshold: float
    early_stopping_patience: Optional[int]
    early_stopping_min_delta: float
    max_grad_norm: Optional[float]
    num_params: int
    trainable_params: int
    architecture: str


def prepare_tiles_and_splits(
    config: dict,
    resolved: ResolvedTrainingPaths,
    max_tiles: Optional[int] = None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    all_tiles = load_filtered_tiles(resolved.filtered_tiles_path)
    if max_tiles is not None:
        viz_config = config.get("visualization", {})
        rep_tile_ids = get_representative_tile_ids_for_viz(
            viz_config, resolved.path_key, resolved.tile_size
        )
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

    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    test_split = config["data"]["test_split"]
    validate_data_splits(train_split, val_split, test_split)
    train_tiles, val_tiles, test_tiles = create_data_splits(
        all_tiles,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )
    logger.info(f"Train: {len(train_tiles)}, Val: {len(val_tiles)}, Test: {len(test_tiles)}")

    use_bg_aug = config["data"].get("use_background_and_augmentation", False)
    extended_path = resolved.filtered_tiles_path.parent / "extended_training_tiles.json"
    if use_bg_aug and extended_path.exists():
        train_tiles, ext_config, ext_stats = load_extended_training_tiles(extended_path)
        logger.info(
            f"Loaded extended training set from {extended_path}: {len(train_tiles)} tiles "
            f"(stats: {ext_stats})"
        )
    elif use_bg_aug:
        logger.warning(
            "use_background_and_augmentation is true but extended_training_tiles.json not found at %s; "
            "using filtered_tiles split only. For background + augmentation run prepare_extended_training_set.py.",
            extended_path,
        )

    illumination_filter = config["data"].get("illumination_filter") or "all"
    illumination_include_background = config["data"].get("illumination_include_background", False)
    before_train, before_val, before_test = len(train_tiles), len(val_tiles), len(test_tiles)
    train_tiles, val_tiles, test_tiles = apply_illumination_filter(
        train_tiles, val_tiles, test_tiles,
        illumination_filter, illumination_include_background,
    )
    if illumination_filter in ("shadow", "sun"):
        logger.info(
            f"Illumination filter '{illumination_filter}': train {before_train}->{len(train_tiles)}, "
            f"val {before_val}->{len(val_tiles)}, test {before_test}->{len(test_tiles)}"
            + (" (background included)" if illumination_include_background else " (no background)")
        )

    return train_tiles, val_tiles, test_tiles, all_tiles


def create_training_dataloaders(
    train_tiles: List[dict],
    val_tiles: List[dict],
    resolved: ResolvedTrainingPaths,
    normalization_stats: dict,
    config: dict,
):
    train_augmentation = config["data"].get("augmentation", False)
    augmentation_config = config["data"].get("augmentation_config", {})
    return create_dataloaders(
        train_tiles,
        val_tiles,
        resolved.features_dir,
        resolved.targets_dir,
        normalization_stats,
        batch_size=config["training"]["batch_size"],
        tile_size=resolved.tile_size,
        target_mode=resolved.target_mode,
        binary_threshold=resolved.binary_threshold,
        segmentation_base_dir=resolved.segmentation_dir,
        slope_stripes_base_dir=resolved.slope_stripes_channel_dir,
        use_rgb=resolved.use_rgb,
        use_dem=resolved.use_dem,
        use_slope=resolved.use_slope,
        train_augmentation=train_augmentation,
        augmentation_config=augmentation_config,
    )


def build_model_and_training_components(
    config: dict,
    in_channels: int,
    target_mode: str,
    device: torch.device,
) -> TrainingComponents:
    logger.info("Creating model...")
    model_config = dict(config["model"])
    model_config["in_channels"] = in_channels
    if target_mode == "binary":
        model_config["proximity_max"] = 1
        model_config["output_activation"] = "sigmoid"
    model = create_model(model_config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {trainable_params:,})")

    architecture = config["model"].get("architecture", "unet")
    logger.info(f"Architecture: {architecture}")
    if architecture == "satlaspretrain_unet":
        enc_cfg = config["model"].get("encoder", {})
        logger.info(
            f"Encoder: {enc_cfg.get('name', 'unknown')} "
            f"(pretrained={enc_cfg.get('pretrained', False)}, "
            f"frozen={enc_cfg.get('freeze_encoder', False)})"
        )

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

    lr_scheduler = None
    lr_scheduler_config = config["training"].get("lr_scheduler")
    if lr_scheduler_config and str(lr_scheduler_config).lower() != "none":
        if lr_scheduler_config == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config["training"].get("lr_scheduler_factor", 0.5),
                patience=config["training"].get("lr_scheduler_patience", 10),
                min_lr=config["training"].get("lr_scheduler_min_lr", 1e-6),
            )
            patience = config["training"].get("lr_scheduler_patience", 10)
            factor = config["training"].get("lr_scheduler_factor", 0.5)
            logger.info(f"Using ReduceLROnPlateau scheduler (patience={patience}, factor={factor})")
        else:
            logger.warning(f"Unknown LR scheduler '{lr_scheduler_config}', ignoring")
    else:
        logger.info("No LR scheduler (fixed learning rate)")

    early_stopping_patience = config["training"].get("early_stopping_patience")
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    if early_stopping_patience:
        logger.info(
            f"Early stopping enabled (patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta})"
        )

    max_grad_norm = config["training"].get("max_grad_norm")
    if max_grad_norm:
        logger.info(f"Gradient clipping enabled (max_norm={max_grad_norm})")

    return TrainingComponents(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        iou_threshold=iou_threshold,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        max_grad_norm=max_grad_norm,
        num_params=num_params,
        trainable_params=trainable_params,
        architecture=architecture,
    )
