import pytest
import torch

from src.models.architectures import _bound_proximity, UNet
from src.models.factory import create_model


class TestBoundProximity:
    def test_clamp_keeps_in_range(self):
        x = torch.tensor([[-5.0, 0.0], [10.0, 25.0]])
        out = _bound_proximity(x, 20.0, "clamp", 0.3)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0
        assert out[0, 0].item() == 0.0
        assert out[0, 1].item() == 0.0
        assert out[1, 0].item() == 10.0
        assert out[1, 1].item() == 20.0

    def test_sigmoid_in_range(self):
        x = torch.tensor([[-100.0], [0.0], [100.0]])
        out = _bound_proximity(x, 20.0, "sigmoid", 0.3)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0
        assert out[1].item() == pytest.approx(10.0, abs=0.1)

    def test_sigmoid_steep_in_range(self):
        x = torch.randn(4, 1, 8, 8) * 10
        out = _bound_proximity(x, 20.0, "sigmoid_steep", 0.3)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0

    def test_clamp_with_max_zero(self):
        x = torch.tensor([[-1.0], [1.0]])
        out = _bound_proximity(x, 0.0, "clamp", 0.3)
        assert out.shape == x.shape
        assert (out == 0.0).all()


class TestUNetOutputInRange:
    def test_clamp_output_in_0_20(self):
        model = UNet(
            in_channels=5,
            out_channels=1,
            base_channels=16,
            dropout=0.0,
            proximity_max=20.0,
            output_activation="clamp",
        )
        x = torch.randn(2, 5, 64, 64) * 10
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0

    def test_sigmoid_output_in_0_20(self):
        model = UNet(
            in_channels=5,
            out_channels=1,
            base_channels=16,
            dropout=0.0,
            proximity_max=20.0,
            output_activation="sigmoid",
        )
        x = torch.randn(2, 5, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0

    def test_proximity_max_zero_no_bound(self):
        model = UNet(
            in_channels=5,
            out_channels=1,
            base_channels=16,
            dropout=0.0,
            proximity_max=0,
        )
        x = torch.randn(2, 5, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)
        assert out.min().item() != out.max().item()


class TestFactoryOutputActivation:
    def test_factory_clamp(self):
        config = {
            "architecture": "unet",
            "in_channels": 5,
            "out_channels": 1,
            "base_channels": 16,
            "dropout": 0.0,
            "proximity_max": 20,
            "output_activation": "clamp",
        }
        model = create_model(config)
        x = torch.randn(1, 5, 64, 64) * 5
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 20.0
