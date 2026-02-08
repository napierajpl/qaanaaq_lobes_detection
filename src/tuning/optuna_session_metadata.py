from pathlib import Path

from src.utils.path_utils import resolve_path
from src.utils.proximity_utils import infer_proximity_token


def current_session_metadata(base_config: dict, mode: str, project_root: Path) -> dict:
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
