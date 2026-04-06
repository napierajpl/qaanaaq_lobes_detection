"""
End-to-end tests: minimal data preparation, HP tuning, and training.

Requires dev data to exist (run `poetry run python scripts/prepare_training_data.py --dev` once).
Run with: pytest tests/e2e/ -m e2e -v
Skip by default: pytest (without -m e2e) does not run these.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.conftest import dev_data_ready, make_minimal_config

pytestmark = pytest.mark.e2e


def _run_script(project_root: Path, script: str, *args, timeout: int = 600):
    cmd = [sys.executable, script] + list(args)
    return subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def project_root():
    from src.utils.path_utils import get_project_root
    return get_project_root(Path(__file__).resolve())


class TestE2EMinimalTraining:
    """Run training for 1 epoch on dev data and check MLflow run and metrics."""

    def test_minimal_training_run_and_metrics(self, project_root, tmp_path):
        if not dev_data_ready(project_root, tile_size=256):
            pytest.skip(
                "Dev data (256) not found. Run: poetry run python scripts/prepare_training_data.py --dev"
            )
        config_path = make_minimal_config(project_root, tmp_path / "training_config_e2e.yaml")

        result = _run_script(
            project_root,
            "scripts/train_model.py",
            "--config", str(config_path),
            "--dev",
            timeout=300,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)

        import mlflow
        mlflow.set_tracking_uri(project_root / "mlruns")
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("lobe_detection")
        assert exp is not None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=20,
            order_by=["attributes.start_time DESC"],
        )
        assert len(runs) >= 1
        run = next((r for r in runs if "best_val_loss" in (r.data.metrics or {})), None)
        assert run is not None, (
            "No recent run with best_val_loss. Runs may be in progress or ordering changed."
        )
        metrics = run.data.metrics
        assert "best_val_loss" in metrics
        assert isinstance(metrics["best_val_loss"], (int, float))
        assert metrics["best_val_loss"] > 0


class TestE2EMinimalTuning:
    """Run one Optuna trial (1 epoch) on dev data and check study result."""

    def test_minimal_tuning_one_trial(self, project_root, tmp_path):
        if not dev_data_ready(project_root, tile_size=256):
            pytest.skip(
                "Dev data (256) not found. Run: poetry run python scripts/prepare_training_data.py --dev"
            )
        config_path = make_minimal_config(project_root, tmp_path / "tuning_config_e2e.yaml")
        csv_path = tmp_path / "e2e_trials.csv"

        result = _run_script(
            project_root,
            "scripts/tune_hyperparameters.py",
            "--config", str(config_path),
            "--dev",
            "--n-trials", "1",
            "--no-seed",
            "--results-csv", str(csv_path),
            timeout=600,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)

        assert csv_path.exists()
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "Expected at least one trial row"
        row = rows[0]
        assert row.get("state") in ("COMPLETE", "PRUNED")
        if row.get("state") == "COMPLETE":
            val = float(row.get("value", 0))
            assert val > 0


class TestE2EDataPrepThenTraining:
    """Run data preparation (--dev) then minimal training. Skips if raw data missing."""

    def test_data_prep_then_training(self, project_root, tmp_path):
        result = _run_script(
            project_root,
            "scripts/prepare_training_data.py",
            "--dev",
            timeout=600,
        )
        if result.returncode != 0:
            pytest.skip(
                "Data preparation failed (missing raw data?). "
                "Ensure data/raw/ rasters exist; see README."
            )
        if not dev_data_ready(project_root, tile_size=256):
            pytest.skip("Dev data not produced by prepare_training_data")
        config_path = make_minimal_config(project_root, tmp_path / "training_config_e2e.yaml")
        result2 = _run_script(
            project_root,
            "scripts/train_model.py",
            "--config", str(config_path),
            "--dev",
            timeout=300,
        )
        assert result2.returncode == 0, (result2.stdout + result2.stderr)
        import mlflow
        mlflow.set_tracking_uri(project_root / "mlruns")
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("lobe_detection")
        assert exp is not None
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
        assert len(runs) >= 1 and "best_val_loss" in runs[0].data.metrics
