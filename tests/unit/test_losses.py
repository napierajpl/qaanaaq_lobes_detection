import pytest
import torch
import torch.nn as nn

from src.models.losses import (
    ACLLoss,
    BCEWithLabelSmoothing,
    CombinedLoss,
    DiceLoss,
    EncouragementLoss,
    FocalLoss,
    IoULoss,
    SmoothL1Loss,
    SoftIoULoss,
    WeightedSmoothL1Loss,
)


@pytest.fixture
def batch_2d():
    torch.manual_seed(42)
    return torch.rand(2, 1, 16, 16)


@pytest.fixture
def binary_pred():
    torch.manual_seed(42)
    return torch.rand(2, 1, 16, 16)


@pytest.fixture
def binary_target():
    torch.manual_seed(99)
    return torch.rand(2, 1, 16, 16).round()


@pytest.fixture
def proximity_pred():
    torch.manual_seed(42)
    return torch.rand(2, 1, 16, 16) * 20.0


@pytest.fixture
def proximity_target():
    torch.manual_seed(99)
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:8, 4:8] = 10.0
    return t


class TestBCEWithLabelSmoothing:
    def test_output_is_scalar(self, binary_pred, binary_target):
        loss_fn = BCEWithLabelSmoothing(smoothing=0.1)
        loss = loss_fn(binary_pred, binary_target)
        assert loss.shape == ()

    def test_zero_smoothing_matches_plain_bce(self, binary_pred, binary_target):
        loss_smooth = BCEWithLabelSmoothing(smoothing=0.0)
        loss_plain = nn.BCELoss(reduction="mean")
        result_smooth = loss_smooth(binary_pred, binary_target)
        result_plain = loss_plain(binary_pred, binary_target)
        assert result_smooth.item() == pytest.approx(result_plain.item(), abs=1e-6)

    def test_smoothing_reduces_loss_on_confident_targets(self):
        pred = torch.tensor([0.99, 0.01])
        target = torch.tensor([1.0, 0.0])
        loss_no_smooth = BCEWithLabelSmoothing(smoothing=0.0)(pred, target)
        loss_smoothed = BCEWithLabelSmoothing(smoothing=0.1)(pred, target)
        assert loss_smoothed.item() > loss_no_smooth.item()

    def test_smoothing_changes_effective_targets(self):
        pred = torch.tensor([0.8, 0.2])
        target = torch.tensor([1.0, 0.0])
        loss_no_smooth = BCEWithLabelSmoothing(smoothing=0.0)(pred, target)
        loss_smoothed = BCEWithLabelSmoothing(smoothing=0.1)(pred, target)
        assert loss_smoothed.item() != loss_no_smooth.item()


class TestSmoothL1Loss:
    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = SmoothL1Loss(beta=1.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()

    def test_zero_loss_when_pred_equals_target(self, proximity_target):
        loss_fn = SmoothL1Loss(beta=1.0)
        loss = loss_fn(proximity_target, proximity_target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_loss_when_pred_differs(self, proximity_pred, proximity_target):
        loss_fn = SmoothL1Loss(beta=1.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.item() > 0.0


class TestWeightedSmoothL1Loss:
    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = WeightedSmoothL1Loss()
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()

    def test_zero_loss_when_pred_equals_target(self, proximity_target):
        loss_fn = WeightedSmoothL1Loss()
        loss = loss_fn(proximity_target, proximity_target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_higher_lobe_weight_increases_loss(self, proximity_pred, proximity_target):
        loss_low = WeightedSmoothL1Loss(lobe_weight=1.0)(proximity_pred, proximity_target)
        loss_high = WeightedSmoothL1Loss(lobe_weight=10.0)(proximity_pred, proximity_target)
        assert loss_high.item() > loss_low.item()

    def test_weights_applied_only_above_threshold(self):
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4) * 10.0
        loss_w1 = WeightedSmoothL1Loss(lobe_weight=1.0, lobe_threshold=5.0)(pred, target)
        loss_w5 = WeightedSmoothL1Loss(lobe_weight=5.0, lobe_threshold=5.0)(pred, target)
        assert loss_w5.item() == pytest.approx(loss_w1.item() * 5.0, rel=1e-5)


class TestDiceLoss:
    def test_output_in_zero_one(self, proximity_pred, proximity_target):
        loss_fn = DiceLoss(threshold=5.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_prediction_near_zero(self, proximity_target):
        loss_fn = DiceLoss(threshold=5.0)
        loss = loss_fn(proximity_target, proximity_target)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_prediction_near_one(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss_fn = DiceLoss(threshold=5.0)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = DiceLoss(threshold=5.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()


class TestIoULoss:
    def test_output_in_zero_one(self, proximity_pred, proximity_target):
        loss_fn = IoULoss(threshold=5.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_prediction_near_zero(self, proximity_target):
        loss_fn = IoULoss(threshold=5.0)
        loss = loss_fn(proximity_target, proximity_target)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_prediction_near_one(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss_fn = IoULoss(threshold=5.0)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_iou_worse_than_dice_for_partial_overlap(self):
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, :4, :] = 10.0
        pred = torch.zeros(1, 1, 8, 8)
        pred[:, :, 2:6, :] = 10.0
        iou_loss = IoULoss(threshold=5.0)(pred, target)
        dice_loss = DiceLoss(threshold=5.0)(pred, target)
        assert iou_loss.item() >= dice_loss.item()


class TestSoftIoULoss:
    def test_output_in_zero_one(self, proximity_pred, proximity_target):
        loss_fn = SoftIoULoss(threshold=5.0)
        loss = loss_fn(proximity_pred, proximity_target)
        assert 0.0 <= loss.item() <= 1.0

    def test_matching_inputs_give_low_loss(self):
        x = torch.ones(1, 1, 8, 8) * 10.0
        loss_fn = SoftIoULoss(threshold=5.0)
        loss = loss_fn(x, x)
        assert loss.item() < 0.1

    def test_opposite_inputs_give_high_loss(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8) * 20.0
        loss_fn = SoftIoULoss(threshold=5.0)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.5

    def test_is_differentiable(self):
        pred = (torch.rand(1, 1, 8, 8) * 10.0).requires_grad_(True)
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss = SoftIoULoss(threshold=5.0)(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestEncouragementLoss:
    def test_zero_when_no_lobes(self):
        pred = torch.rand(1, 1, 8, 8) * 3.0
        target = torch.zeros(1, 1, 8, 8)
        loss = EncouragementLoss(lobe_threshold=5.0)(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_zero_when_predictions_match_targets(self):
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss = EncouragementLoss(lobe_threshold=5.0)(target, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_when_underpredicting_lobes(self):
        pred = torch.ones(1, 1, 8, 8) * 2.0
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss = EncouragementLoss(lobe_threshold=5.0)(pred, target)
        assert loss.item() > 0.0

    def test_penalty_increases_with_encouragement_weight(self):
        pred = torch.ones(1, 1, 8, 8) * 2.0
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss_low = EncouragementLoss(lobe_threshold=5.0, encouragement_weight=1.0)(pred, target)
        loss_high = EncouragementLoss(lobe_threshold=5.0, encouragement_weight=5.0)(pred, target)
        assert loss_high.item() > loss_low.item()

    def test_is_differentiable(self):
        pred = (torch.rand(1, 1, 8, 8) * 3.0).requires_grad_(True)
        target = torch.ones(1, 1, 8, 8) * 10.0
        loss = EncouragementLoss(lobe_threshold=5.0)(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestFocalLoss:
    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = FocalLoss()
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()

    def test_positive_loss_for_mismatched(self, proximity_pred, proximity_target):
        loss_fn = FocalLoss()
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.item() > 0.0

    def test_higher_gamma_focuses_on_hard_examples(self, proximity_pred, proximity_target):
        loss_g0 = FocalLoss(gamma=0.0)(proximity_pred, proximity_target)
        loss_g2 = FocalLoss(gamma=2.0)(proximity_pred, proximity_target)
        assert loss_g2.item() < loss_g0.item()

    def test_alpha_affects_class_weighting(self, proximity_pred, proximity_target):
        loss_low_alpha = FocalLoss(alpha=0.25, gamma=0.0)(proximity_pred, proximity_target)
        loss_high_alpha = FocalLoss(alpha=0.75, gamma=0.0)(proximity_pred, proximity_target)
        assert loss_low_alpha.item() != loss_high_alpha.item()

    def test_reduction_none_returns_full_tensor(self, proximity_pred, proximity_target):
        loss_fn = FocalLoss(reduction="none")
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == proximity_pred.shape

    def test_reduction_sum(self, proximity_pred, proximity_target):
        loss_fn = FocalLoss(reduction="sum")
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()
        loss_none = FocalLoss(reduction="none")(proximity_pred, proximity_target)
        assert loss.item() == pytest.approx(loss_none.sum().item(), rel=1e-5)


class TestCombinedLoss:
    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = CombinedLoss()
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()

    def test_weights_sum_correctly(self, proximity_pred, proximity_target):
        iou_only = CombinedLoss(iou_weight=1.0, regression_weight=0.0)
        reg_only = CombinedLoss(iou_weight=0.0, regression_weight=1.0)
        combined = CombinedLoss(iou_weight=0.3, regression_weight=0.7)
        iou_val = iou_only(proximity_pred, proximity_target)
        reg_val = reg_only(proximity_pred, proximity_target)
        combined_val = combined(proximity_pred, proximity_target)
        expected = 0.3 * iou_val.item() + 0.7 * reg_val.item()
        assert combined_val.item() == pytest.approx(expected, rel=1e-5)

    def test_soft_iou_variant(self, proximity_pred, proximity_target):
        loss_hard = CombinedLoss(use_soft_iou=False)(proximity_pred, proximity_target)
        loss_soft = CombinedLoss(use_soft_iou=True)(proximity_pred, proximity_target)
        assert loss_hard.item() != loss_soft.item()

    def test_positive_loss(self, proximity_pred, proximity_target):
        loss = CombinedLoss()(proximity_pred, proximity_target)
        assert loss.item() > 0.0


class TestACLLoss:
    def test_output_is_scalar(self, proximity_pred, proximity_target):
        loss_fn = ACLLoss()
        loss = loss_fn(proximity_pred, proximity_target)
        assert loss.shape == ()

    def test_positive_loss(self, proximity_pred, proximity_target):
        loss = ACLLoss()(proximity_pred, proximity_target)
        assert loss.item() > 0.0

    def test_lambda_weights_components(self, proximity_pred, proximity_target):
        loss_dice_only = ACLLoss(acl_lambda=1.0)(proximity_pred, proximity_target)
        loss_focal_only = ACLLoss(acl_lambda=0.0)(proximity_pred, proximity_target)
        loss_balanced = ACLLoss(acl_lambda=0.5)(proximity_pred, proximity_target)
        expected = 0.5 * loss_dice_only.item() + 0.5 * loss_focal_only.item()
        assert loss_balanced.item() == pytest.approx(expected, rel=1e-4)

    def test_is_differentiable(self):
        pred = (torch.rand(1, 1, 8, 8) * 20.0).requires_grad_(True)
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 2:6, 2:6] = 10.0
        loss = ACLLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
