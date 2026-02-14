#!/usr/bin/env python3
"""
Hyperparameter tuning using Optuna.

This script performs automated hyperparameter optimization for the lobe detection model.
"""

import logging
import sys
import datetime as dt
import uuid
import warnings
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path
from src.tuning.optuna_csv import (
    load_rows,
    row_value,
    row_dt,
    compatibility_mismatches,
    enqueue_seed_from_row,
    append_study_trials_csv,
    append_rows_from_existing_csv,
)
from src.tuning.optuna_session_metadata import current_session_metadata
from src.tuning.optuna_prompts import prompt_seed_choice_single_file
from src.tuning.optuna_best_params import save_best_params
from src.tuning.optuna_plots import update_progress_plot
from scripts.train_model import train_model_with_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: dict, mode: str) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary
        mode: "dev" or "production"

    Returns:
        Best validation loss (to minimize)
    """
    # Create a copy of config to modify
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy

    # Suggest hyperparameters
    # Top 5 most impactful (Q1: Start with 5)
    # Optuna v4+: prefer suggest_float(..., log=True)
    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    config["training"]["focal_alpha"] = trial.suggest_float("focal_alpha", 0.25, 0.95)
    config["training"]["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 4.0)

    # Q2: Include loss function and encoder name
    loss_function = trial.suggest_categorical(
        "loss_function",
        ["focal", "combined", "weighted_smooth_l1"]
    )
    config["training"]["loss_function"] = loss_function

    encoder_name = trial.suggest_categorical(
        "encoder_name",
        ["resnet50", "resnet152", "swin_v2_base", "swin_v2_tiny"]
    )
    config["model"]["encoder"]["name"] = encoder_name

    # Additional hyperparameters (medium priority)
    config["model"]["decoder_dropout"] = trial.suggest_float("decoder_dropout", 0.0, 0.5)
    config["training"]["lr_scheduler_patience"] = trial.suggest_int("lr_scheduler_patience", 5, 20)
    config["training"]["lr_scheduler_factor"] = trial.suggest_float("lr_scheduler_factor", 0.1, 0.9)
    config["training"]["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 10.0, log=True)
    config["model"]["encoder"]["unfreeze_after_epoch"] = trial.suggest_int("unfreeze_after_epoch", 0, 50)

    # Set MLflow experiment name for hyperparameter tuning
    config["mlflow"]["experiment_name"] = "lobe_detection_hp_tuning"

    # Run training and return best validation loss
    # Q3: Optimize validation loss (recommendation)
    try:
        lr = config["training"]["learning_rate"]
        wd = config["training"]["weight_decay"]
        run_name = f"optuna_t{trial.number:03d}_{encoder_name}_{loss_function}_lr{lr:.1e}_wd{wd:.1e}"

        best_val_loss = train_model_with_config(
            config=config,
            mode=mode,
            trial=trial,
            run_name=run_name,
        )
        # Keep Optuna trial outcome visible at the end (short, single line).
        print(f"[OPTUNA][DONE] trial={trial.number} best_val_loss={best_val_loss:.6f}", flush=True)
        return best_val_loss
    except optuna.TrialPruned:
        # train_model_with_config prints the MLflow run id at prune time.
        print(f"[OPTUNA][PRUNED] trial={trial.number}", flush=True)
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        # Return a high loss value to indicate failure
        return float("inf")


def main():
    """Main function for hyperparameter tuning."""
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "training_config.yaml",
        help="Path to base training config file",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev tiles instead of production",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,  # Q4: Estimated for 8 hours
        help="Number of trials to run (default: 30)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="lobe_detection_hp_tuning",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: in-memory)",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        default=True,
        help="Enable pruning (default: True)",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Where to write Optuna results CSV (default: data/optuna_results/<study>_trials.csv)",
    )
    parser.add_argument(
        "--seed-from-previous-best",
        action="store_true",
        default=True,
        help="Seed first trial from previous best in CSV (default: True)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        default=False,
        help="Disable seeding from previous best",
    )
    parser.add_argument(
        "--import-from",
        type=Path,
        action="append",
        default=[],
        help="Append rows from an existing tuning CSV into the single history CSV, then exit. "
             "Example: --import-from data/optuna_results/lobe_detection_hp_tuning_production.csv",
    )

    args = parser.parse_args()

    # Reduce log noise from known, non-actionable warnings
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The verbose parameter is deprecated\..*",
        category=UserWarning,
    )

    # Load base config
    config_path = resolve_path(args.config, project_root)
    with open(config_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    mode = "dev" if args.dev else "production"
    session_meta = current_session_metadata(base_config, mode, project_root)

    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Pruning: {args.pruning}")
    logger.info("")

    # Create study
    # Q5: MedianPruner (recommendation)
    pruner = optuna.pruners.MedianPruner() if args.pruning else None

    # Q4: TPESampler (recommendation)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",  # Minimize validation loss
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,  # Resume if study exists
    )

    # Determine CSV path (no DB: CSV is our “memory”)
    if args.results_csv is None:
        results_dir = project_root / "data" / "optuna_results"
        results_csv = results_dir / f"{args.study_name}_trials.csv"
    else:
        results_dir = args.results_csv.parent
        results_csv = args.results_csv

    # Progress plot (updated after each trial); optional per-epoch plot when MLflow tracking is file
    progress_plot_path = results_dir / f"{args.study_name}_progress.png"
    tracking_uri = base_config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        base = tracking_uri.replace("file:", "").rstrip("/")
        if not Path(base).is_absolute():
            tracking_uri = f"file:{str(project_root / base)}"
    update_progress_plot(study, progress_plot_path, tracking_uri=tracking_uri)
    print(
        f"[RESULTS] csv_path={results_csv}",
        flush=True,
    )
    print(
        f"[PROGRESS] plot_path={progress_plot_path}",
        flush=True,
    )
    epochs_plot_path = results_dir / f"{args.study_name}_progress_epochs.png"
    print(
        f"[PROGRESS] epochs_plot_path={epochs_plot_path} (updated after each trial, all epochs)",
        flush=True,
    )

    # Explicit, user-driven import of existing CSV(s) into the single history file
    if args.import_from:
        total = 0
        for src in args.import_from:
            src_path = resolve_path(src, project_root)
            n = append_rows_from_existing_csv(
                src_csv=src_path,
                dst_csv=results_csv,
                default_study_name=args.study_name,
            )
            print(f"[IMPORT] source_csv={src_path} appended_rows={n}", flush=True)
            total += n
        print(f"[IMPORT] done total_appended_rows={total} destination_csv={results_csv}", flush=True)
        return

    # Session identifier for append-only CSV
    session_started_at = dt.datetime.now().isoformat(timespec="seconds")
    session_id = f"{session_started_at.replace(':','').replace('-','')}_{uuid.uuid4().hex[:8]}"

    # Seed from previous best by default (unless explicitly disabled)
    seed_summary = {
        "strategy": "none",
        "source_csv": str(results_csv),
        "trial_number": "",
        "value": "",
        "params": {},
        "session_id": "",
    }

    if args.no_seed:
        seed_summary["strategy"] = "disabled"
    elif args.seed_from_previous_best:
        rows = load_rows(results_csv)
        # Filter to same study + mode
        eligible = [
            r for r in rows
            if (r.get("study_name") == args.study_name and r.get("mode") == mode)
        ]
        completed = [r for r in eligible if r.get("state") == "COMPLETE"]

        recent_best_row = None
        if eligible:
            recent_session_id = max(eligible, key=row_dt).get("session_id", "")
            recent_rows = [r for r in completed if r.get("session_id") == recent_session_id]
            if recent_rows:
                recent_best_row = min(recent_rows, key=row_value)

        compatible_completed = [r for r in completed if not compatibility_mismatches(r, session_meta)]
        compatible_best_row = min(compatible_completed, key=row_value) if compatible_completed else None

        if recent_best_row is None:
            if compatible_best_row is not None:
                seed_params = enqueue_seed_from_row(study, compatible_best_row)
                seed_summary.update(
                    {
                        "strategy": "best_compatible",
                        "trial_number": str(compatible_best_row.get("trial_number", "")),
                        "value": str(compatible_best_row.get("value", "")),
                        "params": seed_params,
                        "session_id": str(compatible_best_row.get("session_id", "")),
                    }
                )
        else:
            mismatches = compatibility_mismatches(recent_best_row, session_meta)
            if not mismatches:
                seed_params = enqueue_seed_from_row(study, recent_best_row)
                seed_summary.update(
                    {
                        "strategy": "recent_session",
                        "trial_number": str(recent_best_row.get("trial_number", "")),
                        "value": str(recent_best_row.get("value", "")),
                        "params": seed_params,
                        "session_id": str(recent_best_row.get("session_id", "")),
                    }
                )
            else:
                if not sys.stdin.isatty():
                    # No interactive prompt → pick safest option
                    if compatible_best_row is not None:
                        seed_params = enqueue_seed_from_row(study, compatible_best_row)
                        seed_summary.update(
                            {
                                "strategy": "best_compatible",
                                "trial_number": str(compatible_best_row.get("trial_number", "")),
                                "value": str(compatible_best_row.get("value", "")),
                                "params": seed_params,
                                "session_id": str(compatible_best_row.get("session_id", "")),
                            }
                        )
                else:
                    choice = prompt_seed_choice_single_file(mismatches, recent_best_row, compatible_best_row)
                    if choice == "use_recent":
                        seed_params = enqueue_seed_from_row(study, recent_best_row)
                        seed_summary.update(
                            {
                                "strategy": "forced_recent",
                                "trial_number": str(recent_best_row.get("trial_number", "")),
                                "value": str(recent_best_row.get("value", "")),
                                "params": seed_params,
                                "session_id": str(recent_best_row.get("session_id", "")),
                            }
                        )
                    elif choice == "use_compatible" and compatible_best_row is not None:
                        seed_params = enqueue_seed_from_row(study, compatible_best_row)
                        seed_summary.update(
                            {
                                "strategy": "best_compatible",
                                "trial_number": str(compatible_best_row.get("trial_number", "")),
                                "value": str(compatible_best_row.get("value", "")),
                                "params": seed_params,
                                "session_id": str(compatible_best_row.get("session_id", "")),
                            }
                        )

    # Always print a single, copy/paste-friendly seed line at session start.
    print(
        "[SEED] "
        f"strategy={seed_summary['strategy']} "
        f"source_csv={seed_summary['source_csv'] or 'N/A'} "
        f"from_session_id={seed_summary.get('session_id') or 'N/A'} "
        f"trial_number={seed_summary['trial_number'] or 'N/A'} "
        f"value={seed_summary['value'] or 'N/A'} "
        f"params={seed_summary['params'] or {}} "
        f"current_session_id={session_id}",
        flush=True,
    )

    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    logger.info("")

    def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        update_progress_plot(study, progress_plot_path, tracking_uri=tracking_uri)

    study.optimize(
        lambda trial: objective(trial, base_config, mode),
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[_progress_callback],
    )

    # Append this session's trials to the single history CSV
    append_study_trials_csv(
        study=study,
        csv_path=results_csv,
        session_meta=session_meta,
        session_id=session_id,
        session_started_at=session_started_at,
    )
    logger.info(f"Optuna trials appended to: {results_csv}")

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    complete_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"Number of complete trials: {complete_count}")
    logger.info("")

    if complete_count > 0:
        logger.info("Best trial:")
        best_trial = study.best_trial
        logger.info(f"  Value (validation loss): {best_trial.value:.6f}")
        logger.info("  Params:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
        logger.info("")
        save_best_params(project_root, study)
        logger.info("To retrain with best hyperparameters, update training_config.yaml")
        logger.info("or use the saved parameters from best_hyperparameters.yaml")
    else:
        logger.info("No completed trials; best_hyperparameters.yaml not updated.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
