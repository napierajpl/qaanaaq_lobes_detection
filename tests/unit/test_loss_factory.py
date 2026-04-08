import pytest
import torch
import torch.nn as nn

from src.training.loss_factory import create_criterion, _LOSS_REGISTRY


SUPPORTED_LOSS_NAMES = list(_LOSS_REGISTRY.keys())


class TestCreateCriterionSupported:
    @pytest.fixture
    def base_config(self):
        return {"loss_function": "smooth_l1", "iou_threshold": 5.0}

    @pytest.mark.parametrize("loss_name", SUPPORTED_LOSS_NAMES)
    def test_returns_nn_module(self, loss_name):
        config = {"loss_function": loss_name, "iou_threshold": 5.0}
        criterion = create_criterion(config)
        assert isinstance(criterion, nn.Module)

    @pytest.mark.parametrize("loss_name", SUPPORTED_LOSS_NAMES)
    def test_callable_produces_scalar(self, loss_name):
        config = {
            "loss_function": loss_name,
            "iou_threshold": 5.0,
            "bce_label_smoothing": 0.1,
        }
        criterion = create_criterion(config)
        if loss_name == "bce":
            pred = torch.rand(2, 1, 8, 8)
            target = torch.rand(2, 1, 8, 8)
        else:
            pred = torch.rand(2, 1, 8, 8) * 10
            target = torch.rand(2, 1, 8, 8) * 10
        loss = criterion(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_bce_without_smoothing(self):
        config = {"loss_function": "bce", "bce_label_smoothing": 0.0}
        criterion = create_criterion(config)
        assert isinstance(criterion, nn.BCELoss)

    def test_bce_with_smoothing(self):
        from src.models.losses import BCEWithLabelSmoothing

        config = {"loss_function": "bce", "bce_label_smoothing": 0.1}
        criterion = create_criterion(config)
        assert isinstance(criterion, BCEWithLabelSmoothing)

    def test_bce_with_pos_weight(self):
        from src.models.losses import BCELossWithPosWeight

        config = {"loss_function": "bce", "bce_pos_weight": 5.0}
        criterion = create_criterion(config)
        assert isinstance(criterion, BCELossWithPosWeight)
        pred = torch.full((2, 1, 4, 4), 0.5)
        target = torch.zeros((2, 1, 4, 4))
        target[:, :, :2, :] = 1.0
        loss = criterion(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


class TestCreateCriterionUnknown:
    def test_unknown_loss_raises(self):
        config = {"loss_function": "nonexistent_loss"}
        with pytest.raises(ValueError, match="Unknown loss function"):
            create_criterion(config)

    def test_missing_loss_function_raises(self):
        with pytest.raises(ValueError, match="must contain 'loss_function'"):
            create_criterion({})


class TestCreateCriterionBinaryMode:
    def test_dice_binary_uses_low_threshold(self):
        from src.models.losses import DiceLoss

        config = {"loss_function": "dice", "iou_threshold": 5.0}
        criterion = create_criterion(config, target_mode="binary")
        assert isinstance(criterion, DiceLoss)
        assert criterion.threshold == pytest.approx(0.5)

    def test_dice_proximity_uses_config_threshold(self):
        from src.models.losses import DiceLoss

        config = {"loss_function": "dice", "iou_threshold": 7.0}
        criterion = create_criterion(config, target_mode="proximity")
        assert isinstance(criterion, DiceLoss)
        assert criterion.threshold == pytest.approx(7.0)

    def test_non_dice_binary_mode_ignored(self):
        from src.models.losses import IoULoss

        config = {"loss_function": "iou", "iou_threshold": 5.0}
        criterion = create_criterion(config, target_mode="binary")
        assert isinstance(criterion, IoULoss)
        assert criterion.threshold == pytest.approx(5.0)
