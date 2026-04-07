"""
Factory for creating loss functions from training config.
"""

import logging
from typing import Any, Dict

import torch.nn as nn

from src.models.losses import (
    SmoothL1Loss,
    WeightedSmoothL1Loss,
    DiceLoss,
    IoULoss,
    SoftIoULoss,
    EncouragementLoss,
    FocalLoss,
    CombinedLoss,
    ACLLoss,
)

logger = logging.getLogger(__name__)


def _get_iou_threshold(training_config: Dict[str, Any], default: float = 5.0) -> float:
    return float(training_config.get("iou_threshold", default))


def _create_smooth_l1(_config: Dict[str, Any]) -> nn.Module:
    return SmoothL1Loss()


def _create_weighted_smooth_l1(config: Dict[str, Any]) -> nn.Module:
    lobe_weight = config.get("lobe_weight", 5.0)
    lobe_threshold = config.get("iou_threshold", 5.0)
    return WeightedSmoothL1Loss(lobe_weight=lobe_weight, lobe_threshold=lobe_threshold)


def _create_dice(config: Dict[str, Any]) -> nn.Module:
    threshold = _get_iou_threshold(config)
    return DiceLoss(threshold=threshold)


def _create_iou(config: Dict[str, Any]) -> nn.Module:
    threshold = _get_iou_threshold(config)
    return IoULoss(threshold=threshold)


def _create_soft_iou(config: Dict[str, Any]) -> nn.Module:
    threshold = _get_iou_threshold(config)
    return SoftIoULoss(threshold=threshold)


def _create_encouragement(config: Dict[str, Any]) -> nn.Module:
    threshold = _get_iou_threshold(config)
    encouragement_weight = config.get("encouragement_weight", 2.0)
    return EncouragementLoss(
        lobe_threshold=threshold,
        encouragement_weight=encouragement_weight,
    )


def _create_focal(config: Dict[str, Any]) -> nn.Module:
    alpha = config.get("focal_alpha", 0.25)
    gamma = config.get("focal_gamma", 2.0)
    threshold = _get_iou_threshold(config)
    return FocalLoss(alpha=alpha, gamma=gamma, lobe_threshold=threshold)


def _create_combined(config: Dict[str, Any]) -> nn.Module:
    iou_weight = config.get("iou_weight", 0.5)
    regression_weight = config.get("regression_weight", 0.5)
    threshold = _get_iou_threshold(config)
    lobe_weight = config.get("lobe_weight", 5.0)
    use_soft_iou = config.get("use_soft_iou", False)
    return CombinedLoss(
        iou_weight=iou_weight,
        regression_weight=regression_weight,
        iou_threshold=threshold,
        lobe_weight=lobe_weight,
        lobe_threshold=threshold,
        use_soft_iou=use_soft_iou,
    )


def _create_acl(config: Dict[str, Any]) -> nn.Module:
    acl_lambda = config.get("acl_lambda", 0.5)
    threshold = _get_iou_threshold(config)
    focal_alpha = config.get("focal_alpha", 0.75)
    focal_gamma = config.get("focal_gamma", 2.0)
    return ACLLoss(
        acl_lambda=acl_lambda,
        threshold=threshold,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )


def _create_bce(config: Dict[str, Any]) -> nn.Module:
    import torch
    from src.models.losses import BCEWithLabelSmoothing
    smoothing = config.get("bce_label_smoothing") or 0.0
    smoothing = float(smoothing)
    pos_weight = config.get("bce_pos_weight")
    if pos_weight is not None:
        pw = torch.tensor([float(pos_weight)])
        logger.info("BCE with pos_weight=%.2f", float(pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if smoothing <= 0:
        return nn.BCELoss()
    return BCEWithLabelSmoothing(smoothing=smoothing)


_LOSS_REGISTRY: Dict[str, Any] = {
    "smooth_l1": _create_smooth_l1,
    "weighted_smooth_l1": _create_weighted_smooth_l1,
    "dice": _create_dice,
    "iou": _create_iou,
    "soft_iou": _create_soft_iou,
    "encouragement": _create_encouragement,
    "focal": _create_focal,
    "combined": _create_combined,
    "acl": _create_acl,
    "bce": _create_bce,
}


def create_criterion(
    training_config: Dict[str, Any],
    target_mode: str = "proximity",
) -> nn.Module:
    """
    Create loss criterion from training config.

    Args:
        training_config: config["training"] (must contain "loss_function")
        target_mode: "proximity" | "binary". For binary, dice uses threshold 0.5.

    Returns:
        Initialized loss module

    Raises:
        ValueError: If loss_function is unknown
    """
    name = training_config.get("loss_function")
    if not name:
        raise ValueError("training_config must contain 'loss_function'")
    name = str(name).strip().lower()
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss function: {name}. Supported: {list(_LOSS_REGISTRY.keys())}"
        )
    mode = (target_mode or "proximity").lower()
    if mode == "binary" and name == "dice":
        config_override = {**training_config, "iou_threshold": 0.5}
        criterion = _LOSS_REGISTRY[name](config_override)
    else:
        criterion = _LOSS_REGISTRY[name](training_config)
    logger.info("Using %s", _criterion_description(name, training_config))
    return criterion


def _criterion_description(name: str, config: Dict[str, Any]) -> str:
    threshold = _get_iou_threshold(config)
    if name == "smooth_l1":
        return "SmoothL1Loss"
    if name == "weighted_smooth_l1":
        return (
            f"WeightedSmoothL1Loss(lobe_weight={config.get('lobe_weight', 5.0)}, "
            f"lobe_threshold={config.get('iou_threshold', 5.0)})"
        )
    if name == "dice":
        return f"DiceLoss(threshold={threshold})"
    if name == "iou":
        return f"IoULoss(threshold={threshold})"
    if name == "soft_iou":
        return f"SoftIoULoss(threshold={threshold})"
    if name == "encouragement":
        return (
            f"EncouragementLoss(threshold={threshold}, "
            f"encouragement_weight={config.get('encouragement_weight', 2.0)})"
        )
    if name == "focal":
        return (
            f"FocalLoss(alpha={config.get('focal_alpha', 0.25)}, "
            f"gamma={config.get('focal_gamma', 2.0)}, lobe_threshold={threshold})"
        )
    if name == "combined":
        u = " (with soft IoU)" if config.get("use_soft_iou") else ""
        return (
            f"CombinedLoss (IoU + Weighted Smooth L1){u} "
            f"(iou_weight={config.get('iou_weight', 0.5)}, "
            f"regression_weight={config.get('regression_weight', 0.5)}, threshold={threshold})"
        )
    if name == "acl":
        return (
            f"ACLLoss(acl_lambda={config.get('acl_lambda', 0.5)}, threshold={threshold}, "
            f"focal_alpha={config.get('focal_alpha', 0.75)}, focal_gamma={config.get('focal_gamma', 2.0)})"
        )
    if name == "bce":
        smooth = config.get("bce_label_smoothing") or 0.0
        if float(smooth) > 0:
            return f"BCEWithLabelSmoothing(smoothing={smooth})"
        return "BCELoss"
    return name
