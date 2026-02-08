import logging
from pathlib import Path

import optuna
import yaml

logger = logging.getLogger(__name__)


def save_best_params(project_root: Path, study: optuna.Study) -> None:
    best_trial = study.best_trial
    best_params_path = project_root / "configs" / "best_hyperparameters.yaml"
    best_params = {
        "best_validation_loss": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "hyperparameters": dict(best_trial.params),
    }
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)
    logger.info(f"Best hyperparameters saved to: {best_params_path}")
