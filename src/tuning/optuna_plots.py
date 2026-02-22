from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna


def update_progress_plot(
    study: optuna.Study,
    path: Path,
    tracking_uri: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot trial number vs value and best-so-far; save to path.
    If tracking_uri is set, also fetch per-epoch val_loss from MLflow and save
    an all-epochs plot to path.stem + "_epochs" + path.suffix.
    Returns the path to the epochs plot if created, else None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        fig, ax = plt.subplots()
        ax.set_xlabel("Trial")
        ax.set_ylabel("Value (validation loss)")
        ax.set_title("HP tuning progress")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        if tracking_uri:
            _save_epochs_plot(study, path, tracking_uri)
        return None
    by_number = sorted(completed, key=lambda t: t.number)
    xs = [t.number for t in by_number]
    ys = [t.value for t in by_number]
    best_so_far = []
    b = float("inf")
    for t in by_number:
        if t.value < b:
            b = t.value
        best_so_far.append(b)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, alpha=0.7, label="Trial value")
    ax.plot(xs, best_so_far, color="C1", label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Value (validation loss)")
    ax.set_title("HP tuning progress")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    epochs_path = None
    if tracking_uri:
        epochs_path = _save_epochs_plot(study, path, tracking_uri)
    return epochs_path


def _save_epochs_plot(study: optuna.Study, progress_path: Path, tracking_uri: str) -> Optional[Path]:
    """Fetch val_loss per epoch from MLflow for each trial; plot all on one figure. Returns path if saved."""
    try:
        from mlflow import MlflowClient
    except ImportError:
        return None
    client = MlflowClient(tracking_uri=tracking_uri)
    x_labels = []
    y_values = []
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.number,
    )
    for t in completed:
        run_id = (t.user_attrs or {}).get("mlflow_run_id")
        if not run_id:
            continue
        try:
            history = client.get_metric_history(run_id, "val_loss")
        except Exception:
            continue
        if not history:
            continue
        by_step = sorted(history, key=lambda m: m.step)
        for m in by_step:
            x_labels.append(f"T{t.number} E{m.step}")
            y_values.append(m.value)
    if not y_values:
        return None
    epochs_path = progress_path.parent / f"{progress_path.stem}_epochs{progress_path.suffix}"
    fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 0.15), 5))
    x_pos = list(range(len(y_values)))
    ax.plot(x_pos, y_values, "b-", alpha=0.8, linewidth=0.8)
    ax.set_ylabel("Val loss")
    ax.set_title("HP tuning: validation loss at every epoch (all trials)")
    ax.set_ylim(bottom=0)
    if len(x_labels) <= 40:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        step = max(1, len(x_labels) // 20)
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(epochs_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return epochs_path
