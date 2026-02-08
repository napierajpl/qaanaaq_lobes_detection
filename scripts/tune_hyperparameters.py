#!/usr/bin/env python3
"""
Hyperparameter tuning using Optuna.

This script performs automated hyperparameter optimization for the lobe detection model.
"""

import logging
import sys
import csv
import datetime as dt
import uuid
import warnings
from pathlib import Path
from typing import Optional

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path
from src.utils.proximity_utils import infer_proximity_token

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

TRIALS_CSV_HEADER = [
    "exported_at",
    "session_id",
    "session_started_at",
    "study_name",
    "mode",
    "trial_number",
    "state",
    "value",
    "datetime_start",
    "datetime_complete",
] + TUNED_PARAM_KEYS + [
    "mlflow_experiment_id",
    "mlflow_run_id",
    "mlflow_run_name",
    "mlflow_tracking_uri",
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
    "pruned_epoch",
    "pruned_val_loss",
    "best_val_loss",
    "best_val_mae",
    "best_val_iou",
]


def _current_session_metadata(base_config: dict, mode: str) -> dict:
    """Metadata used to judge whether seeding from previous best makes sense."""
    project_root = get_project_root(Path(__file__))
    tile_size = base_config["data"].get("tile_size", 256)
    path_key = mode if tile_size == 256 else f"{mode}_512"
    paths = base_config["paths"][path_key]
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
        "proximity_token": infer_proximity_token(targets_dir),
        "mlflow_tracking_uri": str(base_config.get("mlflow", {}).get("tracking_uri", "")),
    }


def _load_previous_best(csv_path: Path) -> Optional[dict]:
    """
    Return the best COMPLETED trial row from a CSV, or None.

    Note: this helper is still used for small one-off CSVs; for the new single-file
    append-only workflow we use `_load_rows()` + selection logic in `main()`.
    """
    if not csv_path.exists():
        return None
    rows = _load_rows(csv_path)
    if not rows:
        return None
    completed = [r for r in rows if (r.get("state") == "COMPLETE")]
    if not completed:
        return None
    return min(completed, key=lambda r: _row_value(r))


def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _row_value(r: dict) -> float:
    try:
        return float(r.get("value", "inf"))
    except Exception:
        return float("inf")


def _row_dt(r: dict) -> dt.datetime:
    """
    Best-effort timestamp for sorting rows newest-first.
    Prefers `session_started_at` (new format), then exported_at, then datetime_complete.
    """
    for k in ("session_started_at", "exported_at", "datetime_complete", "datetime_start"):
        v = (r.get(k) or "").strip()
        if not v:
            continue
        try:
            # Accept both "2026-01-27T20:13:47" and full iso with microseconds
            return dt.datetime.fromisoformat(v)
        except Exception:
            continue
    return dt.datetime.min


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


def _enqueue_seed_from_row(study: optuna.Study, best_row: dict) -> dict:
    """Enqueue the previous best hyperparameters as the first trial. Returns params used."""
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
    return params


def _prompt_seed_choice_single_file(
    mismatches: dict,
    recent_best_row: dict,
    compatible_best_row: Optional[dict],
) -> str:
    """
    Ask user what to do when the most recent session best is incompatible.
    Returns one of: "use_recent", "use_compatible", "no_seed".
    """
    print("")
    print("WARNING: Most recent session best hyperparameters may be incompatible with this session.")
    print(
        "Most recent session best: "
        f"trial_number={recent_best_row.get('trial_number','?')} value={recent_best_row.get('value','?')} "
        f"session_id={recent_best_row.get('session_id','?')}"
    )
    print("Mismatches:")
    for k, v in mismatches.items():
        print(f"  - {k}: previous={v['previous']}  current={v['current']}")
    print("")

    options: list[tuple[str, str]] = []
    options.append(("1", "Use most recent anyway (may make no sense)"))
    if compatible_best_row is not None:
        options.append(
            (
                "2",
                "Use best compatible from history: "
                f"trial_number={compatible_best_row.get('trial_number','?')} value={compatible_best_row.get('value','?')} "
                f"session_id={compatible_best_row.get('session_id','?')}",
            )
        )
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

    if compatible_best_row is not None:
        if answer == "1":
            return "use_recent"
        if answer == "2":
            return "use_compatible"
        return "no_seed"
    else:
        if answer == "1":
            return "use_recent"
        return "no_seed"


def _append_study_trials_csv(
    study: optuna.Study,
    csv_path: Path,
    session_meta: dict,
    session_id: str,
    session_started_at: str,
) -> None:
    """
    Append this session's trials to a single CSV (append-only history).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exported_at = dt.datetime.now().isoformat(timespec="seconds")
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIALS_CSV_HEADER)
        if not file_exists:
            writer.writeheader()

        for t in study.trials:
            ua = t.user_attrs or {}
            row = {
                "exported_at": exported_at,
                "session_id": session_id,
                "session_started_at": session_started_at,
                "study_name": study.study_name,
                "mode": session_meta.get("mode", ""),
                "trial_number": t.number,
                "state": t.state.name,
                "value": t.value if t.value is not None else "",
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else "",
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else "",
            }

            for k in TUNED_PARAM_KEYS:
                row[k] = t.params.get(k, "")

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

            for k in (
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


def _append_rows_from_existing_csv(
    src_csv: Path,
    dst_csv: Path,
    default_study_name: str,
) -> int:
    """
    Append rows from an existing Optuna CSV into our single append-only history CSV.
    This is explicit (user-driven) migration, not automatic bootstrap.
    """
    rows = _load_rows(src_csv)
    if not rows:
        return 0

    exported_at = (rows[0].get("exported_at") or dt.datetime.now().isoformat(timespec="seconds")).strip()
    session_id = f"import_{src_csv.stem}_{uuid.uuid4().hex[:8]}"

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = dst_csv.exists() and dst_csv.stat().st_size > 0
    written = 0

    with open(dst_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIALS_CSV_HEADER)
        if not file_exists:
            writer.writeheader()

        for r in rows:
            out = {k: "" for k in TRIALS_CSV_HEADER}
            out.update(r)
            out["session_id"] = r.get("session_id") or session_id
            out["session_started_at"] = r.get("session_started_at") or exported_at
            out["study_name"] = r.get("study_name") or default_study_name
            writer.writerow(out)
            written += 1

    return written


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
        results_csv = results_dir / f"{args.study_name}_trials.csv"
    else:
        results_dir = args.results_csv.parent
        results_csv = args.results_csv

    # Always print where we read/write tuning results
    print(
        f"[RESULTS] csv_path={results_csv}",
        flush=True,
    )

    # Explicit, user-driven import of existing CSV(s) into the single history file
    if args.import_from:
        total = 0
        for src in args.import_from:
            src_path = resolve_path(src, project_root)
            n = _append_rows_from_existing_csv(
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
        rows = _load_rows(results_csv)
        # Filter to same study + mode
        eligible = [
            r for r in rows
            if (r.get("study_name") == args.study_name and r.get("mode") == mode)
        ]
        completed = [r for r in eligible if r.get("state") == "COMPLETE"]

        recent_best_row = None
        if eligible:
            recent_session_id = max(eligible, key=_row_dt).get("session_id", "")
            recent_rows = [r for r in completed if r.get("session_id") == recent_session_id]
            if recent_rows:
                recent_best_row = min(recent_rows, key=_row_value)

        compatible_completed = [r for r in completed if not _compatibility_mismatches(r, session_meta)]
        compatible_best_row = min(compatible_completed, key=_row_value) if compatible_completed else None

        if recent_best_row is None:
            if compatible_best_row is not None:
                seed_params = _enqueue_seed_from_row(study, compatible_best_row)
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
            mismatches = _compatibility_mismatches(recent_best_row, session_meta)
            if not mismatches:
                seed_params = _enqueue_seed_from_row(study, recent_best_row)
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
                        seed_params = _enqueue_seed_from_row(study, compatible_best_row)
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
                    choice = _prompt_seed_choice_single_file(mismatches, recent_best_row, compatible_best_row)
                    if choice == "use_recent":
                        seed_params = _enqueue_seed_from_row(study, recent_best_row)
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
                        seed_params = _enqueue_seed_from_row(study, compatible_best_row)
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

    study.optimize(
        lambda trial: objective(trial, base_config, mode),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Append this session's trials to the single history CSV
    _append_study_trials_csv(
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

    _save_best_params(project_root, study)
    logger.info("")
    logger.info("To retrain with best hyperparameters, update training_config.yaml")
    logger.info("or use the saved parameters from best_hyperparameters.yaml")


def _save_best_params(project_root: Path, study: optuna.Study) -> None:
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


if __name__ == "__main__":
    main()
