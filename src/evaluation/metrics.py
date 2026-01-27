"""
Evaluation metrics for model performance.
"""

import torch


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        MAE value
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        RMSE value
    """
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 5.0) -> float:
    """
    Compute Intersection over Union for lobe pixels.

    Args:
        pred: Predicted values
        target: Target values
        threshold: Threshold to consider as lobe (default: 8.0)

    Returns:
        IoU value
    """
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()
