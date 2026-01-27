#!/usr/bin/env python3
"""
Hyperparameter tuning using Optuna.

This script performs automated hyperparameter optimization for the lobe detection model.
"""

import logging
import sys
import csv
import datetime as dt
from pathlib import Path
from typing import Optional

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path

# Import after path setup
from scripts.train_model import train_model_with_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TUNED_PARAM_KEYS = [
    "learning_rate",
    "batch_size",
    "weight_decay",
    "focal_alpha",
    "focal_gamma",
    "loss_function",
    "encoder_name",
    "decoder_dropout",
    "lr_scheduler_patience",
    "lr_scheduler_factor",
    "max_grad_norm",
    "unfreeze_after_epoch",
]


def _infer_proximity_token(targets_dir: str) -> str:
    s = targets_dir.lower()
    if "proximity20" in s:
        return "proximity20"
    if "proximity10" in s:
        return "proximity10"
    return "unknown"


def _current_session_metadata(base_config: dict, mode: str) -> dict:
    """Metadata used to judge whether seeding from previous best makes sense."""
    project_root = get_project_root(__file__)
    paths = base_config["paths"][mode]
    targets_dir = str(resolve_path(Path(paths["targets_dir"]), project_root))
    filtered_tiles = str(resolve_path(Path(paths["filtered_tiles"]), project_root))
    features_dir = str(resolve_path(Path(paths["features_dir"]), project_root))

    model_cfg = base_config.get("model", {})
    train_cfg = base_config.get("training", {})
    data_cfg = base_config.get("data", {})

    return {
        "mode": mode,
        "model_architecture": str(model_cfg.get("architecture", "")),
        "model_in_channels": int(model_cfg.get("in_channels", 5)),
        "model_out_channels": int(model_cfg.get("out_channels", 1)),
        "training_iou_threshold": float(train_cfg.get("iou_threshold", 5.0)),
        "data_normalize_rgb": bool(data_cfg.get("normalize_rgb", True)),
        "data_standardize_dem": bool(data_cfg.get("standardize_dem", True)),
        "data_standardize_slope": bool(data_cfg.get("standardize_slope", True)),
        "filtered_tiles_path": filtered_tiles,
        "features_dir": features_dir,
        "targets_dir": targets_dir,
        "proximity_token": _infer_proximity_token(targets_dir),
        "mlflow_tracking_uri": str(base_config.get("mlflow", {}).get("tracking_uri", "")),
    }


def _load_previous_best(csv_path: Path) -> Optional[dict]:
    """Return the best COMPLETED trial row from a previous CSV, or None."""
    if not csv_path.exists():
        return None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    def row_value(r: dict) -> float:
        try:
            return float(r.get("value", "inf"))
        except Exception:
            return float("inf")

    completed = [r for r in rows if (r.get("state") == "COMPLETE")]
    if not completed:
        return None
    return min(completed, key=row_value)


def _compatibility_mismatches(prev: dict, current: dict) -> dict:
    """Return a dict of mismatched metadata keys."""
    mismatches = {}
    keys_to_check = [
        "mode",
        "model_architecture",
        "model_in_channels",
        "model_out_channels",
        "training_iou_threshold",
        "data_normalize_rgb",
        "data_standardize_dem",
        "data_standardize_slope",
        "targets_dir",
        "filtered_tiles_path",
        "proximity_token",
    ]
    for k in keys_to_check:
        prev_v = prev.get(k)
        cur_v = current.get(k)
        if str(prev_v) != str(cur_v):
            mismatches[k] = {"previous": prev_v, "current": cur_v}
    return mismatches


def _prompt_confirm_seed(mismatches: dict, csv_path: Path, prev_trial_number: str) -> bool:
    print("")
    print("WARNING: Previous best hyperparameters may not be compatible with this session.")
    print(f"Source CSV: {csv_path}")
    print(f"Previous best trial number: {prev_trial_number}")
    print("Mismatches:")
    for k, v in mismatches.items():
        print(f"  - {k}: previous={v['previous']}  current={v['current']}")
    print("")
    try:
        answer = input("Use previous best as seed anyway? [y/N]: ").strip().lower()
    except Exception:
        return False
    return answer in ("y", "yes")


def _enqueue_seed_from_row(study: optuna.Study, best_row: dict) -> None:
    """Enqueue the previous best hyperparameters as the first trial."""
    params = {}
    for k in TUNED_PARAM_KEYS:
        if k not in best_row:
            continue
        raw = best_row[k]
        if raw in (None, ""):
            continue
        # basic typing
        if k in ("batch_size", "lr_scheduler_patience", "unfreeze_after_epoch"):
            params[k] = int(float(raw))
        else:
            # loss_function / encoder_name are strings, everything else float
            if k in ("loss_function", "encoder_name"):
                params[k] = str(raw)
            else:
                params[k] = float(raw)

    if params:
        study.enqueue_trial(params)


def _archive_csv_path(latest_csv_path: Path) -> Path:
    """Return a timestamped archive path for a given 'latest' CSV path."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = latest_csv_path.parent / "archive"
    return archive_dir / f"{latest_csv_path.stem}_{ts}{latest_csv_path.suffix}"


def _find_candidate_csvs(results_dir: Path, study_name: str, mode: str) -> list[Path]:
    """Find CSVs for this study/mode, including archives."""
    patterns = [
        f"{study_name}_{mode}.csv",
        f"{study_name}_{mode}_*.csv",
        f"archive/{study_name}_{mode}_*.csv",
        f"archive/{study_name}_{mode}.csv",
    ]
    out: list[Path] = []
    for p in patterns:
        out.extend(results_dir.glob(p))
    # Sort newest-first by mtime
    out = [p for p in out if p.exists()]
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # De-dup while preserving order
    seen = set()
    deduped: list[Path] = []
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        deduped.append(p)
    return deduped


def _pick_best_compatible_seed(
    candidate_csvs: list[Path],
    session_meta: dict,
) -> Optional[tuple[Path, dict]]:
    """
    From a list of candidate CSVs, pick the most recent one that has a COMPLETE best row
    and no compatibility mismatches.
    """
    for csv_path in candidate_csvs:
        best_row = _load_previous_best(csv_path)
        if best_row is None:
            continue
        mismatches = _compatibility_mismatches(best_row, session_meta)
        if not mismatches:
            return (csv_path, best_row)
    return None


def _prompt_seed_choice(
    mismatches: dict,
    latest_csv: Path,
    latest_best_row: dict,
    compatible_pick: Optional[tuple[Path, dict]],
) -> str:
    """
    Ask user what to do when the most recent run seems incompatible.
    Returns one of: "use_latest", "use_archive", "no_seed".
    """
    print("")
    print("WARNING: Most recent previous best hyperparameters may be incompatible with this session.")
    print(f"Most recent CSV: {latest_csv}")
    print(f"Most recent best trial: {latest_best_row.get('trial_number', '?')} value={latest_best_row.get('value', '?')}")
    print("Mismatches:")
    for k, v in mismatches.items():
        print(f"  - {k}: previous={v['previous']}  current={v['current']}")
    print("")

    options: list[tuple[str, str]] = []
    options.append(("1", "Use most recent run anyway (may make no sense)"))
    if compatible_pick is not None:
        p, row = compatible_pick
        options.append(("2", f"Use archived compatible run: {p} (trial={row.get('trial_number','?')} value={row.get('value','?')})"))
        options.append(("3", "No seeding (cold start)"))
    else:
        options.append(("2", "No seeding (cold start)"))

    print("Choose seeding strategy:")
    for key, label in options:
        print(f"  {key}) {label}")
    try:
        answer = input("Enter choice: ").strip()
    except Exception:
        return "no_seed"

    if compatible_pick is not None:
        if answer == "1":
            return "use_latest"
        if answer == "2":
            return "use_archive"
        return "no_seed"
    else:
        if answer == "1":
            return "use_latest"
        return "no_seed"


def _export_study_csv(study: optuna.Study, csv_path: Path, session_meta: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a consistent header (trial info + tuned params + metadata + mlflow ids)
    header = [
        "exported_at",
        "study_name",
        "trial_number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
    ] + TUNED_PARAM_KEYS + [
        # MLflow fields (from trial user_attrs if available)
        "mlflow_experiment_id",
        "mlflow_run_id",
        "mlflow_run_name",
        "mlflow_tracking_uri",
        # Compatibility fields
        "mode",
        "model_architecture",
        "model_in_channels",
        "model_out_channels",
        "training_iou_threshold",
        "data_normalize_rgb",
        "data_standardize_dem",
        "data_standardize_slope",
        "filtered_tiles_path",
        "features_dir",
        "targets_dir",
        "proximity_token",
        # prune info
        "pruned_epoch",
        "pruned_val_loss",
        # best metrics (if recorded)
        "best_val_loss",
        "best_val_mae",
        "best_val_iou",
    ]

    exported_at = dt.datetime.now().isoformat(timespec="seconds")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for t in study.trials:
            row = {
                "exported_at": exported_at,
                "study_name": study.study_name,
                "trial_number": t.number,
                "state": t.state.name,
                "value": t.value if t.value is not None else "",
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else "",
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else "",
            }
            for k in TUNED_PARAM_KEYS:
                row[k] = t.params.get(k, "")

            # user attrs (set by train_model_with_config)
            ua = t.user_attrs or {}
            for k in (
                "mlflow_experiment_id",
                "mlflow_run_id",
                "mlflow_run_name",
                "mlflow_tracking_uri",
                "pruned_epoch",
                "pruned_val_loss",
                "best_val_loss",
                "best_val_mae",
                "best_val_iou",
            ):
                row[k] = ua.get(k, "")

            # session meta (ensure present even if trial did not populate user_attrs)
            for k in (
                "mode",
                "model_architecture",
                "model_in_channels",
                "model_out_channels",
                "training_iou_threshold",
                "data_normalize_rgb",
                "data_standardize_dem",
                "data_standardize_slope",
                "filtered_tiles_path",
                "features_dir",
                "targets_dir",
                "proximity_token",
            ):
                row[k] = ua.get(k, session_meta.get(k, ""))

            writer.writerow(row)


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
    
    project_root = get_project_root(__file__)
    
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
        help="Where to write Optuna results CSV (default: data/optuna_results/<study>_<mode>.csv)",
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
    
    args = parser.parse_args()
    
    # Load base config
    config_path = resolve_path(args.config, project_root)
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    mode = "dev" if args.dev else "production"
    session_meta = _current_session_metadata(base_config, mode)
    
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
        results_csv = results_dir / f"{args.study_name}_{mode}.csv"
    else:
        results_dir = args.results_csv.parent
        results_csv = args.results_csv

    # Seed from previous best by default (unless explicitly disabled)
    if (args.seed_from_previous_best and not args.no_seed):
        # 1) Most recent (latest) CSV for this study/mode
        latest_best = _load_previous_best(results_csv)

        # 2) Find older/archived candidates for fallback
        candidates = _find_candidate_csvs(results_dir, args.study_name, mode)
        compatible_pick = _pick_best_compatible_seed(candidates, session_meta)

        if latest_best is None:
            if compatible_pick is None:
                logger.info(f"No previous best found (latest or archive) for seeding. Starting without seed.")
            else:
                seed_csv, seed_row = compatible_pick
                _enqueue_seed_from_row(study, seed_row)
                logger.info(f"Seeding enabled: using archived compatible run: {seed_csv}")
        else:
            mismatches = _compatibility_mismatches(latest_best, session_meta)
            if not mismatches:
                _enqueue_seed_from_row(study, latest_best)
                logger.info(f"Seeding enabled: using most recent best from {results_csv}")
            else:
                # If we can't prompt (e.g. no stdin), default to no seed unless user forces it via input.
                if not sys.stdin.isatty():
                    logger.warning("Most recent seed looks incompatible and no TTY available; skipping seeding.")
                else:
                    choice = _prompt_seed_choice(mismatches, results_csv, latest_best, compatible_pick)
                    if choice == "use_latest":
                        _enqueue_seed_from_row(study, latest_best)
                        logger.info("Seeding enabled: using most recent (forced by user).")
                    elif choice == "use_archive" and compatible_pick is not None:
                        seed_csv, seed_row = compatible_pick
                        _enqueue_seed_from_row(study, seed_row)
                        logger.info(f"Seeding enabled: using archived compatible run: {seed_csv}")
                    else:
                        logger.info("Seeding disabled: cold start selected.")
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    logger.info("")
    
    study.optimize(
        lambda trial: objective(trial, base_config, mode),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Export rich CSV for transparency / future seeding
    _export_study_csv(study, results_csv, session_meta)
    logger.info(f"Optuna results CSV written to: {results_csv}")

    # Also keep a timestamped archive copy so we can seed from older compatible sessions.
    archive_csv = _archive_csv_path(results_csv)
    _export_study_csv(study, archive_csv, session_meta)
    logger.info(f"Optuna results CSV archived to: {archive_csv}")
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logger.info(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    logger.info("")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value (validation loss): {best_trial.value:.6f}")
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    logger.info("")
    logger.info("=" * 80)
    
    # Save best parameters to file (real YAML)
    best_params_path = project_root / "configs" / "best_hyperparameters.yaml"
    best_params = {
        "best_validation_loss": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "hyperparameters": dict(best_trial.params),
    }

    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)
    
    logger.info(f"Best hyperparameters saved to: {best_params_path}")
    logger.info("")
    logger.info("To retrain with best hyperparameters, update training_config.yaml")
    logger.info("or use the saved parameters from best_hyperparameters.yaml")


if __name__ == "__main__":
    main()
