"""CLI builder for training script: data-driven argument definitions."""

from pathlib import Path
from typing import Any, List

TRAIN_MODEL_ARG_SPECS: List[dict] = [
    {
        "option": "--config",
        "type": Path,
        "default_path": ("configs", "training_config.yaml"),
        "help": "Path to training config file",
    },
    {
        "option": "--dev",
        "action": "store_true",
        "help": "Use dev tiles (cropped 1024x1024) instead of full dataset",
    },
    {
        "option": "--mode",
        "type": str,
        "choices": ["dev", "production", "synthetic_parenthesis"],
        "default": None,
        "help": "Dataset mode: dev, production, or synthetic_parenthesis (sanity-check). Default: dev if --dev else production.",
    },
    {
        "option": "--run-name",
        "type": str,
        "default": None,
        "help": "MLflow run name (default: auto-generated)",
    },
    {
        "option": "--max-epochs",
        "type": int,
        "default": None,
        "help": "Override num_epochs from config (e.g. 1 for a quick dry run)",
    },
    {
        "option": "--tile-size",
        "type": int,
        "default": None,
        "choices": [256, 512],
        "help": "Override data tile size from config (e.g. 256 for dev when dev/train_512 does not exist)",
    },
    {
        "option": "--best-hparams",
        "action": "store_true",
        "help": "Override config with best hyperparameters from configs/best_hyperparameters.yaml",
    },
    {
        "option": "--best-hparams-path",
        "type": Path,
        "default": None,
        "help": "Path to best hyperparameters YAML (default: configs/best_hyperparameters.yaml)",
    },
    {
        "option": "--hp_from_run_id",
        "type": str,
        "default": None,
        "metavar": "RUN_ID",
        "help": "Apply hyperparameters from an MLflow run ID (e.g. from MLflow UI).",
    },
    {
        "option": "--max-tiles",
        "type": int,
        "default": None,
        "metavar": "N",
        "help": "Max tiles to use in total (before train/val/test split). Use for quick runs.",
    },
    {
        "option": "--filtered-tiles",
        "type": Path,
        "default": None,
        "metavar": "PATH",
        "help": "Override filtered_tiles.json path (e.g. subset with targets only).",
    },
    {
        "option": "--illumination-filter",
        "type": str,
        "choices": ["all", "sun", "shadow"],
        "default": None,
        "help": "Train only on sun or shadow tiles (plus background). Requires illumination tags from add_illumination_tags.py.",
    },
    {
        "option": "--enable-layer",
        "nargs": "+",
        "metavar": "NAME",
        "default": None,
        "help": "Enable specific layers by name (e.g. --enable-layer dem slope).",
    },
    {
        "option": "--disable-layer",
        "nargs": "+",
        "metavar": "NAME",
        "default": None,
        "help": "Disable specific layers by name (e.g. --disable-layer segmentation).",
    },
    {
        "option": "--resume",
        "type": Path,
        "default": None,
        "metavar": "PATH",
        "help": "Resume training from a full checkpoint (training_latest.pt or best_model.pt with training_loop_state).",
    },
    {
        "option": "--init-weights",
        "type": Path,
        "default": None,
        "metavar": "PATH",
        "help": (
            "Initialize model weights from a checkpoint (best_model.pt or training_latest.pt) "
            "but start a fresh training loop (new optimizer, epoch 1, reset early stopping). "
            "Use to fine-tune from a previous run with different settings."
        ),
    },
]


def build_train_parser(project_root: Path) -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN model for lobe detection")
    for spec in TRAIN_MODEL_ARG_SPECS:
        spec = dict(spec)
        default_path = spec.pop("default_path", None)
        if default_path is not None:
            spec["default"] = project_root.joinpath(*default_path)
        option = spec.pop("option")
        parser.add_argument(option, **spec)
    return parser


def apply_cli_overrides(config: dict, args: Any) -> None:
    """Apply CLI argument overrides to config. Mutates config in place."""
    if getattr(args, "max_epochs", None) is not None:
        config["training"]["num_epochs"] = args.max_epochs
    if getattr(args, "tile_size", None) is not None:
        config["data"]["tile_size"] = args.tile_size
    if getattr(args, "illumination_filter", None) is not None:
        config["data"]["illumination_filter"] = args.illumination_filter

    enable = getattr(args, "enable_layer", None) or []
    disable = getattr(args, "disable_layer", None) or []
    layers = config.get("layers", {})
    for name in enable:
        if name in layers:
            layers[name]["enabled"] = True
        else:
            raise ValueError(f"Unknown layer '{name}'. Available: {list(layers.keys())}")
    for name in disable:
        if name in layers:
            layers[name]["enabled"] = False
        else:
            raise ValueError(f"Unknown layer '{name}'. Available: {list(layers.keys())}")
