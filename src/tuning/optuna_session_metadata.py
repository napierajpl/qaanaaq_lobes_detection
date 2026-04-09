from pathlib import Path

from src.training.training_config import validate_in_channels
from src.utils.config_utils import get_training_path_key
from src.utils.path_utils import resolve_path
from src.utils.proximity_utils import infer_proximity_token


def current_session_metadata(base_config: dict, mode: str, project_root: Path) -> dict:
    tile_size = base_config["data"].get("tile_size", 256)
    path_key = get_training_path_key(mode, tile_size)
    paths = base_config["paths"][path_key]
    targets_dir = str(resolve_path(Path(paths["targets_dir"]), project_root))
    filtered_tiles = str(resolve_path(Path(paths["filtered_tiles"]), project_root))
    model_cfg = base_config.get("model", {})
    train_cfg = base_config.get("training", {})
    in_channels = validate_in_channels(base_config)
    enabled_layers = [
        name for name, lc in base_config.get("layers", {}).items()
        if lc.get("enabled", True)
    ]
    return {
        "mode": mode,
        "model_architecture": str(model_cfg.get("architecture", "")),
        "model_in_channels": in_channels,
        "model_out_channels": int(model_cfg.get("out_channels", 1)),
        "training_iou_threshold": float(train_cfg.get("iou_threshold", 5.0)),
        "enabled_layers": ", ".join(enabled_layers),
        "filtered_tiles_path": filtered_tiles,
        "targets_dir": targets_dir,
        "proximity_token": infer_proximity_token(targets_dir),
        "mlflow_tracking_uri": str(base_config.get("mlflow", {}).get("tracking_uri", "")),
    }
