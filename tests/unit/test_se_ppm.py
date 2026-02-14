import pytest
import torch

from src.models.se_ppm import SELayer, PyramidPoolingModule


class TestSELayer:
    def test_output_shape_unchanged(self):
        b, c, h, w = 2, 64, 16, 16
        layer = SELayer(c, reduction=16)
        x = torch.randn(b, c, h, w)
        out = layer(x)
        assert out.shape == (b, c, h, w)

    def test_small_channel_uses_safe_reduction(self):
        b, c, h, w = 2, 8, 8, 8
        layer = SELayer(c, reduction=16)
        x = torch.randn(b, c, h, w)
        out = layer(x)
        assert out.shape == (b, c, h, w)


class TestPyramidPoolingModule:
    def test_output_shape_unchanged(self):
        b, c, h, w = 2, 64, 16, 16
        ppm = PyramidPoolingModule(c, bins=(1, 2, 3, 6))
        x = torch.randn(b, c, h, w)
        out = ppm(x)
        assert out.shape == (b, c, h, w)

    def test_different_bins(self):
        b, c, h, w = 2, 128, 32, 32
        ppm = PyramidPoolingModule(c, bins=(1, 2, 4))
        x = torch.randn(b, c, h, w)
        out = ppm(x)
        assert out.shape == (b, c, h, w)


class TestSatlasPretrainUNetWithSEPPM:
    def test_baseline_no_se_ppm_output_shape(self):
        pytest.importorskip("satlaspretrain_models")
        from src.models.factory import create_model

        config = {
            "architecture": "satlaspretrain_unet",
            "in_channels": 5,
            "out_channels": 1,
            "encoder": {"name": "resnet50", "pretrained": False, "freeze_encoder": True},
            "decoder_dropout": 0.2,
            "use_se": False,
            "use_ppm": False,
        }
        model = create_model(config)
        x = torch.randn(1, 5, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert model.ppm is None
        assert model.se is None

    def test_both_se_ppm_output_shape(self):
        pytest.importorskip("satlaspretrain_models")
        from src.models.factory import create_model

        config = {
            "architecture": "satlaspretrain_unet",
            "in_channels": 5,
            "out_channels": 1,
            "encoder": {"name": "resnet50", "pretrained": False, "freeze_encoder": True},
            "decoder_dropout": 0.2,
            "use_se": True,
            "use_ppm": True,
        }
        model = create_model(config)
        x = torch.randn(1, 5, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert model.ppm is not None
        assert model.se is not None

    def test_se_only_output_shape(self):
        pytest.importorskip("satlaspretrain_models")
        from src.models.factory import create_model

        config = {
            "architecture": "satlaspretrain_unet",
            "in_channels": 5,
            "out_channels": 1,
            "encoder": {"name": "resnet50", "pretrained": False, "freeze_encoder": True},
            "decoder_dropout": 0.2,
            "use_se": True,
            "use_ppm": False,
        }
        model = create_model(config)
        x = torch.randn(1, 5, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert model.se is not None
        assert model.ppm is None

    def test_ppm_only_output_shape(self):
        pytest.importorskip("satlaspretrain_models")
        from src.models.factory import create_model

        config = {
            "architecture": "satlaspretrain_unet",
            "in_channels": 5,
            "out_channels": 1,
            "encoder": {"name": "resnet50", "pretrained": False, "freeze_encoder": True},
            "decoder_dropout": 0.2,
            "use_se": False,
            "use_ppm": True,
        }
        model = create_model(config)
        x = torch.randn(1, 5, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert model.ppm is not None
        assert model.se is None
