#!/usr/bin/env python3
"""
Inspect warm_start_metadata.json (saved next to checkpoints each epoch) and optionally resume training.

Example:
  poetry run python scripts/resume_from_saved.py --config configs/training_config.yaml
  poetry run python scripts/resume_from_saved.py --manifest data/models/production/warm_start_metadata.json --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.training.training_config import resolve_training_paths
from src.training.warm_start import WARM_START_MANIFEST_NAME, plot_loss_from_manifest_metrics
from src.utils.path_utils import get_project_root, resolve_path


def _default_manifest_path(project_root: Path, config: dict, mode: str) -> Path:
    resolved = resolve_training_paths(config, mode, project_root, None)
    return resolved.models_dir / WARM_START_MANIFEST_NAME


def main() -> None:
    project_root = get_project_root(Path(__file__))
    parser = argparse.ArgumentParser(
        description="Show warm-start metadata (saved epoch, config snapshot, loss curve) and optionally resume training.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Path to {WARM_START_MANIFEST_NAME} (default: next to models_dir from config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "training_config.yaml",
        help="Training config (must match the architecture and data the checkpoint was trained with).",
    )
    parser.add_argument("--dev", action="store_true", help="Use dev mode paths when resolving default manifest.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "production", "synthetic_parenthesis"],
        default=None,
        help="Dataset mode for default manifest path (default: dev if --dev else production).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open an interactive matplotlib window for the loss preview (if a display is available).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Continue training without confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print info and save loss preview only; do not start training.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config, project_root)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mode = args.mode or ("dev" if args.dev else "production")

    if args.manifest is not None:
        manifest_path = resolve_path(args.manifest, project_root)
    else:
        manifest_path = _default_manifest_path(project_root, config, mode)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Train at least one epoch so training writes warm_start_metadata.json next to models_dir.")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    ckpt_rel = m.get("checkpoint_path")
    ckpt_path = (project_root / ckpt_rel).resolve() if ckpt_rel else None

    print("=== Warm-start / resume ===")
    print(f"Manifest:     {manifest_path}")
    print(f"Last epoch:   {m.get('last_completed_epoch')}")
    print(f"Target epochs (config when saved): {m.get('num_epochs_target')}")
    print(f"Mode (saved): {m.get('mode')}")
    print(f"Best val_loss: {m.get('best_val_loss')}")
    print(f"Best val_mae:  {m.get('best_val_mae')}")
    print(f"Best val_iou:  {m.get('best_val_iou')}")
    print(f"Checkpoint:   {ckpt_path}")
    print(f"Config path (saved): {m.get('config_path')}")
    if m.get("loss_plot_path"):
        print(f"Loss plot (saved path): {m.get('loss_plot_path')}")

    snap = m.get("config_snapshot")
    if isinstance(snap, dict):
        tr = snap.get("training", {})
        print("--- Training params (snapshot) ---")
        print(f"  learning_rate: {tr.get('learning_rate')}")
        print(f"  num_epochs:    {tr.get('num_epochs')}")
        print(f"  loss:          {tr.get('loss_function')}")
        print(f"  batch_size:    {tr.get('batch_size')}")

    metrics_history = m.get("metrics_history") or {}
    preview_path = manifest_path.parent / "warm_start_loss_preview.png"
    if metrics_history.get("epochs") and metrics_history.get("train_loss"):
        fig = plot_loss_from_manifest_metrics(metrics_history, output_path=preview_path)
        if fig is not None:
            import matplotlib.pyplot as plt
            print(f"Loss preview written to: {preview_path}")
            if args.show_plot:
                plt.show()
            plt.close(fig)
    else:
        print("(No metrics_history in manifest; run a newer training job to populate.)")

    if args.dry_run:
        return

    if ckpt_path is None or not ckpt_path.exists():
        print("Checkpoint path missing or file not found; cannot resume.")
        sys.exit(1)

    if not args.yes:
        answer = input("Continue training from this checkpoint? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_model.py"),
        "--config",
        str(config_path),
        "--resume",
        str(ckpt_path),
    ]
    if mode == "dev":
        cmd.append("--dev")
    elif mode == "synthetic_parenthesis":
        cmd.extend(["--mode", "synthetic_parenthesis"])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
