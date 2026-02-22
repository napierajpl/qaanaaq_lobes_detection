import pytest

from src.utils.path_utils import tile_dir_for_pipeline
from src.data_processing.prepare_training_steps import production_steps, dev_steps, PipelineRunner


class TestTileDirForPipeline:
    def test_production_256(self):
        assert tile_dir_for_pipeline(False, 256) == "data/processed/tiles/train"

    def test_production_512(self):
        assert tile_dir_for_pipeline(False, 512) == "data/processed/tiles/train_512"

    def test_dev_256(self):
        assert tile_dir_for_pipeline(True, 256) == "data/processed/tiles/dev/train"

    def test_dev_512(self):
        assert tile_dir_for_pipeline(True, 512) == "data/processed/tiles/dev/train_512"


class TestProductionSteps:
    def test_returns_list_of_tuples(self):
        steps = production_steps(256)
        assert isinstance(steps, list)
        assert len(steps) >= 1
        for desc, cmd in steps:
            assert isinstance(desc, str)
            assert isinstance(cmd, list)
            assert len(cmd) >= 1

    def test_tile_dir_in_feature_and_target_paths(self):
        steps = production_steps(512)
        descs_and_cmds = [c for _, c in steps]
        joined = " ".join(" ".join(c) for c in descs_and_cmds)
        assert "train_512/features" in joined
        assert "train_512/targets" in joined
        assert "train_512/filtered_tiles.json" in joined

    def test_256_uses_train_subdir(self):
        steps = production_steps(256)
        joined = " ".join(" ".join(c) for _, c in steps)
        assert "train/features" in joined
        assert "train/targets" in joined


class TestDevSteps:
    def test_returns_list_of_tuples(self):
        steps = dev_steps(256)
        assert isinstance(steps, list)
        assert len(steps) >= 1
        for desc, cmd in steps:
            assert isinstance(desc, str)
            assert isinstance(cmd, list)

    def test_dev_tile_dir_in_paths(self):
        steps = dev_steps(512)
        joined = " ".join(" ".join(c) for _, c in steps)
        assert "dev/train_512" in joined


class TestPipelineRunner:
    def test_run_steps_records_times(self, tmp_path):
        import sys
        runner = PipelineRunner(tmp_path, dev_mode=False, tile_size=256)
        steps = [("Dummy", [sys.executable, "-c", "print(1)"])]
        runner.run_steps(steps)
        assert len(runner.step_times) == 1
        assert runner.step_times[0][0] == "Dummy"
        assert runner.step_times[0][1] >= 0
