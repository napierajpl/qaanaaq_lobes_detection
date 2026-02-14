"""
Loss functions for training.
"""

import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) for regression."""

    def __init__(self, beta: float = 1.0):
        """
        Initialize Smooth L1 Loss.

        Args:
            beta: Threshold for smooth transition (default: 1.0)
        """
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Smooth L1 Loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Loss value
        """
        return nn.functional.smooth_l1_loss(pred, target, beta=self.beta)


class WeightedSmoothL1Loss(nn.Module):
    """Weighted Smooth L1 Loss to handle class imbalance."""

    def __init__(self, beta: float = 1.0, lobe_weight: float = 5.0, lobe_threshold: float = 5.0):
        """
        Initialize Weighted Smooth L1 Loss.

        Args:
            beta: Threshold for smooth transition
            lobe_weight: Weight for lobe pixels (value >= lobe_threshold)
            lobe_threshold: Threshold to consider as lobe pixel (default: 5.0)
        """
        super().__init__()
        self.beta = beta
        self.lobe_weight = lobe_weight
        self.lobe_threshold = lobe_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted Smooth L1 Loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Loss value
        """
        # Create weight mask: higher weight for lobe pixels
        weights = torch.ones_like(target)
        weights[target >= self.lobe_threshold] = self.lobe_weight

        # Compute element-wise smooth L1 loss
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Smooth L1 formula
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )

        # Apply weights
        weighted_loss = loss * weights

        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""

    def __init__(self, threshold: float = 5.0, smooth: float = 1e-6):
        """
        Initialize Dice Loss.

        Args:
            threshold: Threshold to binarize predictions/targets
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Binarize predictions and targets
        pred_binary = (pred >= self.threshold).float()
        target_binary = (target >= self.threshold).float()

        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class IoULoss(nn.Module):
    """IoU Loss (Jaccard Loss) for segmentation tasks."""

    def __init__(self, threshold: float = 5.0, smooth: float = 1e-6):
        """
        Initialize IoU Loss.

        Args:
            threshold: Threshold to binarize predictions/targets
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU Loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            IoU loss (1 - IoU)
        """
        # Binarize predictions and targets
        pred_binary = (pred >= self.threshold).float()
        target_binary = (target >= self.threshold).float()

        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection

        # IoU with smoothing
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou


class SoftIoULoss(nn.Module):
    """Soft IoU Loss using sigmoid for smooth gradients."""

    def __init__(self, threshold: float = 5.0, temperature: float = 1.0, smooth: float = 1e-6):
        """
        Initialize Soft IoU Loss.

        Args:
            threshold: Threshold to consider as lobe
            temperature: Temperature for sigmoid (higher = sharper transition)
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Soft IoU Loss using sigmoid.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Soft IoU loss (1 - soft IoU)
        """
        # Apply sigmoid to create soft binary masks
        # Shift by threshold so sigmoid(0) = 0.5 when value = threshold
        pred_soft = torch.sigmoid((pred - self.threshold) * self.temperature)
        target_soft = torch.sigmoid((target - self.threshold) * self.temperature)

        # Flatten tensors
        pred_flat = pred_soft.view(-1)
        target_flat = target_soft.view(-1)

        # Compute soft intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection

        # Soft IoU with smoothing
        soft_iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - soft_iou


class EncouragementLoss(nn.Module):
    """Loss that encourages model to predict higher values for lobe areas."""

    def __init__(self, lobe_threshold: float = 5.0, encouragement_weight: float = 2.0):
        """
        Initialize Encouragement Loss.

        Args:
            lobe_threshold: Threshold for lobe pixels
            encouragement_weight: Weight for encouragement term
        """
        super().__init__()
        self.lobe_threshold = lobe_threshold
        self.encouragement_weight = encouragement_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Encouragement Loss.

        Encourages predictions to be >= threshold when targets are >= threshold.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Encouragement loss value
        """
        # Create mask for lobe areas
        lobe_mask = target >= self.lobe_threshold

        if lobe_mask.sum() == 0:
            # No lobes in this batch, return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # For lobe pixels, encourage predictions to be at least threshold
        lobe_preds = pred[lobe_mask]
        lobe_targets = target[lobe_mask]

        # Penalty when prediction < threshold (even if target >= threshold)
        below_threshold = lobe_preds < self.lobe_threshold
        if below_threshold.sum() > 0:
            penalty = torch.mean((self.lobe_threshold - lobe_preds[below_threshold]) ** 2)
        else:
            penalty = torch.tensor(0.0, device=pred.device)

        # Also encourage matching the target value
        mse = torch.mean((lobe_preds - lobe_targets) ** 2)

        return mse + self.encouragement_weight * penalty


class FocalLoss(nn.Module):
    """
    Focal Loss for regression tasks with extreme class imbalance.

    Adapted from the original Focal Loss (Lin et al., 2017) for regression.
    Down-weights easy examples (where prediction is close to target) and focuses
    learning on hard examples (where prediction error is large).

    This is particularly effective for extreme class imbalance (e.g., 93.5% background
    vs 6.5% lobes) where the model tends to minimize loss by predicting background
    everywhere.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lobe_threshold: float = 5.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss for regression.

        Args:
            alpha: Weighting factor for class balancing (0.25 = standard, higher = more weight on lobes)
                  - alpha < 0.5: More weight on background
                  - alpha = 0.5: Balanced
                  - alpha > 0.5: More weight on lobes (recommended for extreme imbalance)
                  - Typical values: 0.25 (standard), 0.75 (strong lobe focus)
            gamma: Focusing parameter (higher = more focus on hard examples)
                  - gamma=0: equivalent to standard MSE
                  - gamma=2.0: standard focal loss (recommended)
                  - gamma>2.0: even more focus on hard examples
            lobe_threshold: Threshold to identify lobe pixels (for alpha weighting)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lobe_threshold = lobe_threshold
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss for regression.

        The loss is computed as:
        FL = alpha * (error_normalized)^gamma * MSE

        Where:
        - error_normalized: Normalized prediction error (0-1 range)
        - (error_normalized)^gamma: Modulating factor (down-weights easy examples)
        - alpha: Class weighting (higher for lobe pixels)

        Args:
            pred: Predicted values [B, C, H, W] or [B, H, W]
            target: Target values [B, C, H, W] or [B, H, W]

        Returns:
            Focal loss value
        """
        # Compute prediction error (absolute difference)
        error = torch.abs(pred - target)

        # Normalize error to [0, 1] range for stable gamma exponentiation
        # Use a reasonable max error (e.g., max of target range)
        # Support both 0-10 and 0-20 proximity map ranges
        max_target_value = target.max().item()
        max_error = max(max_target_value, 20.0)  # Use actual max or 20.0, whichever is larger
        error_normalized = torch.clamp(error / max_error, 0.0, 1.0)

        # Compute base loss (MSE)
        mse = (pred - target) ** 2

        # Compute modulating factor: (error_normalized)^gamma
        # This down-weights easy examples (low error) and focuses on hard examples (high error)
        modulating_factor = error_normalized ** self.gamma

        # Apply alpha weighting for class balancing
        # For lobe pixels: use alpha (higher alpha = more weight on lobes)
        # For background pixels: use (1-alpha) (lower alpha = less weight on background)
        alpha_mask = torch.ones_like(target)
        alpha_mask[target >= self.lobe_threshold] = self.alpha  # Weight for lobe pixels
        alpha_mask[target < self.lobe_threshold] = 1.0 - self.alpha  # Weight for background pixels

        # Compute focal loss: alpha * modulating_factor * mse
        focal_loss = alpha_mask * modulating_factor * mse

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss: IoU + Weighted Smooth L1 for both segmentation and regression."""

    def __init__(
        self,
        iou_weight: float = 0.5,
        regression_weight: float = 0.5,
        iou_threshold: float = 5.0,
        beta: float = 1.0,
        lobe_weight: float = 5.0,
        lobe_threshold: float = 5.0,
        use_soft_iou: bool = False,
    ):
        """
        Initialize Combined Loss.

        Args:
            iou_weight: Weight for IoU loss component
            regression_weight: Weight for regression loss component
            iou_threshold: Threshold for IoU calculation
            beta: Smooth L1 beta parameter
            lobe_weight: Weight for lobe pixels in regression loss
            lobe_threshold: Threshold for lobe pixels
            use_soft_iou: Use soft IoU instead of hard threshold (default: False)
        """
        super().__init__()
        self.iou_weight = iou_weight
        self.regression_weight = regression_weight
        if use_soft_iou:
            self.iou_loss = SoftIoULoss(threshold=iou_threshold)
        else:
            self.iou_loss = IoULoss(threshold=iou_threshold)
        self.regression_loss = WeightedSmoothL1Loss(
            beta=beta, lobe_weight=lobe_weight, lobe_threshold=lobe_threshold
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Combined Loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Combined loss value
        """
        iou_loss = self.iou_loss(pred, target)
        regression_loss = self.regression_loss(pred, target)

        return self.iou_weight * iou_loss + self.regression_weight * regression_loss


class ACLLoss(nn.Module):
    """
    Adaptive Correction Loss (ACL): λ·Dice + (1−λ)·Focal.

    From Gully-ERFNet (Li et al., IJDE 2025) for linear structures, severe imbalance,
    and label noise. Combines overlap (Dice) with hard-example focus (Focal).
    """

    def __init__(
        self,
        acl_lambda: float = 0.5,
        threshold: float = 5.0,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.acl_lambda = acl_lambda
        self.dice_loss = DiceLoss(threshold=threshold, smooth=dice_smooth)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            lobe_threshold=threshold,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.acl_lambda * dice + (1.0 - self.acl_lambda) * focal
