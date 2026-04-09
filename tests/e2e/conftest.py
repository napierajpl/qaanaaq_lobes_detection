import pytest
import yaml
from pathlib import Path

from src.utils.path_utils import get_project_root


@pytest.fixture(scope="module")
def project_root():
    return get_project_root(Path(__file__).resolve())


def dev_data_ready(project_root: Path, tile_size: int = 256) -> bool:
    if tile_size == 256:
        path = project_root / "data/processed/tiles/dev/train/filtered_tiles.json"
    else:
        path = project_root / "data/processed/tiles/dev/train_512/filtered_tiles.json"
    return path.exists()


def make_minimal_config(project_root: Path, out_path: Path) -> Path:
    base = project_root / "configs" / "training_config.yaml"
    with open(base, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data"]["tile_size"] = 256
    config["data"]["use_background_and_augmentation"] = False
    config["data"]["illumination_filter"] = None
    config["layers"]["rgb"]["enabled"] = True
    config["layers"]["dem"]["enabled"] = True
    config["layers"]["slope"]["enabled"] = True
    config["layers"]["segmentation"]["enabled"] = False
    config["layers"]["slope_stripes"]["enabled"] = False
    config["training"]["num_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return out_path
