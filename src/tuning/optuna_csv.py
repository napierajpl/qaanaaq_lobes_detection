import csv
import datetime as dt
import uuid
from pathlib import Path
from typing import Optional

import optuna

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
    "enabled_layers",
    "filtered_tiles_path",
    "targets_dir",
    "proximity_token",
    "pruned_epoch",
    "pruned_val_loss",
    "best_val_loss",
    "best_val_mae",
    "best_val_iou",
]

COMPATIBILITY_KEYS = [
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


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def row_value(r: dict) -> float:
    try:
        return float(r.get("value", "inf"))
    except Exception:
        return float("inf")


def row_dt(r: dict) -> dt.datetime:
    for k in ("session_started_at", "exported_at", "datetime_complete", "datetime_start"):
        v = (r.get(k) or "").strip()
        if not v:
            continue
        try:
            return dt.datetime.fromisoformat(v)
        except Exception:
            continue
    return dt.datetime.min


def load_previous_best(csv_path: Path) -> Optional[dict]:
    if not csv_path.exists():
        return None
    rows = load_rows(csv_path)
    if not rows:
        return None
    completed = [r for r in rows if (r.get("state") == "COMPLETE")]
    if not completed:
        return None
    return min(completed, key=lambda r: row_value(r))


def compatibility_mismatches(prev: dict, current: dict) -> dict:
    mismatches = {}
    for k in COMPATIBILITY_KEYS:
        prev_v = prev.get(k)
        cur_v = current.get(k)
        if str(prev_v) != str(cur_v):
            mismatches[k] = {"previous": prev_v, "current": cur_v}
    return mismatches


def enqueue_seed_from_row(study: optuna.Study, best_row: dict) -> dict:
    params = {}
    for k in TUNED_PARAM_KEYS:
        if k not in best_row:
            continue
        raw = best_row[k]
        if raw in (None, ""):
            continue
        if k in ("batch_size", "lr_scheduler_patience", "unfreeze_after_epoch"):
            params[k] = int(float(raw))
        else:
            if k in ("loss_function", "encoder_name"):
                params[k] = str(raw)
            else:
                params[k] = float(raw)
    if params:
        study.enqueue_trial(params)
    return params


def append_study_trials_csv(
    study: optuna.Study,
    csv_path: Path,
    session_meta: dict,
    session_id: str,
    session_started_at: str,
) -> None:
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
                "enabled_layers",
                "filtered_tiles_path",
                "targets_dir",
                "proximity_token",
            ):
                row[k] = ua.get(k, session_meta.get(k, ""))
            writer.writerow(row)


def append_rows_from_existing_csv(
    src_csv: Path,
    dst_csv: Path,
    default_study_name: str,
) -> int:
    rows = load_rows(src_csv)
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
