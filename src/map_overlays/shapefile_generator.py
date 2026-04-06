"""
Shapefile generation for tile visualization in QGIS.
"""

import logging
from pathlib import Path
from typing import Optional, Set

import geopandas as gpd
from shapely.geometry import box

from src.map_overlays.tile_registry import TileRegistry

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_qml_template(template_name: str, **kwargs: str) -> str:
    template_path = _TEMPLATES_DIR / template_name
    template = template_path.read_text(encoding="utf-8")
    return template.format(**kwargs)


def _train_usage_for_tile(
    tile: dict,
    background_train_ids: Optional[Set[str]] = None,
) -> str:
    """Return train_usage: lobe_train, lobe_val, lobe_test, background_train, or empty."""
    tile_id = tile.get("tile_id", "")
    if background_train_ids and tile_id in background_train_ids:
        return "background_train"
    is_valid = tile.get("filtering", {}).get("is_valid", False)
    if not is_valid:
        return ""
    split_raw = tile.get("split") or ""
    if split_raw == "train":
        return "lobe_train"
    if split_raw == "val":
        return "lobe_val"
    if split_raw == "test":
        return "lobe_test"
    return ""


def generate_tile_index_shapefile(
    registry: TileRegistry,
    output_path: Path,
    label_field: str = "tile_label",
    include_all_tiles: bool = True,
    background_train_ids: Optional[Set[str]] = None,
) -> None:
    """
    Generate shapefile from tile registry for QGIS visualization.

    Args:
        registry: TileRegistry instance
        output_path: Path to output shapefile (without .shp extension)
        label_field: Field name to use for labels in QGIS
        include_all_tiles: If True, include all tiles; if False, only valid tiles
        background_train_ids: If provided, add trn_usage column (lobe_train, lobe_val,
            lobe_test, background_train). Tiles in this set are marked background_train.
    """
    logger.info(f"Generating tile index shapefile: {output_path}")

    # Get tiles
    tiles = registry.get_all_tiles(filter_valid=not include_all_tiles)
    metadata = registry.get_metadata()

    # Create GeoDataFrame
    geometries = []
    attributes = []

    for tile in tiles:
        bounds = tile.get("geographic_bounds", {})
        if not bounds:
            continue

        # Create polygon from bounds
        geom = box(
            bounds["minx"],
            bounds["miny"],
            bounds["maxx"],
            bounds["maxy"],
        )
        geometries.append(geom)

        # Extract tile number for label (e.g., "tile_10020" -> "10020")
        tile_id = tile.get("tile_id", "")
        tile_label = tile_id.replace("tile_", "") if tile_id.startswith("tile_") else tile_id

        split_raw = tile.get("split")
        split_str = split_raw if split_raw else ""

        attrs = {
            "tile_id": tile_id,
            label_field: tile_label,
            "tile_idx": tile.get("tile_idx", -1),
            "tile_size": int(metadata.get("tile_size", 0)),
            "is_valid": int(tile.get("filtering", {}).get("is_valid", False)),
            "rgb_valid": int(tile.get("filtering", {}).get("rgb_valid", False)),
            "has_targets": int(tile.get("filtering", {}).get("has_targets", False)),
            "split": split_str,
        }
        train_usage = _train_usage_for_tile(tile, background_train_ids)
        attrs["trn_usage"] = train_usage

        attributes.append(attrs)

    if not geometries:
        logger.warning("No tiles with geographic bounds found")
        return

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries)

    # Set CRS from metadata
    crs_str = metadata.get("crs")
    if crs_str:
        try:
            gdf.set_crs(crs_str, inplace=True)
        except (ValueError, Exception) as e:
            # CRS setting can fail for various reasons (invalid CRS, geopandas version issues, etc.)
            logger.warning(f"Could not set CRS {crs_str}: {e}")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save shapefile
    gdf.to_file(output_path, driver="ESRI Shapefile")
    logger.info(f"Saved shapefile with {len(gdf)} tiles to {output_path}")

    qml_path = Path(str(output_path).replace(".shp", ".qml"))
    _generate_qml_style_file(qml_path, label_field)
    logger.info(f"Generated QGIS style file: {qml_path}")

    split_qml_path = Path(str(output_path).replace(".shp", "_by_split.qml"))
    _generate_qml_style_file_by_split(split_qml_path, label_field)
    logger.info(f"Generated QGIS style file (by split): {split_qml_path}")

    if "trn_usage" in gdf.columns:
        usage_qml_path = Path(str(output_path).replace(".shp", "_by_train_usage.qml"))
        _generate_qml_style_file_by_train_usage(usage_qml_path, label_field)
        logger.info(f"Generated QGIS style file (by train_usage): {usage_qml_path}")
        usage_counts = gdf["trn_usage"].value_counts()
        logger.info(f"  Train usage: {dict(usage_counts)}")

    # Log summary
    if "is_valid" in gdf.columns:
        valid_count = gdf["is_valid"].sum()
        logger.info(f"  Valid tiles: {valid_count}/{len(gdf)}")

    if "split" in gdf.columns:
        split_counts = gdf["split"].value_counts()
        logger.info(f"  Split distribution: {dict(split_counts)}")


def _generate_qml_style_file(qml_path: Path, label_field: str) -> None:
    qml_content = _load_qml_template("style_default.qml", label_field=label_field)
    qml_path.parent.mkdir(parents=True, exist_ok=True)
    qml_path.write_text(qml_content, encoding="utf-8")


def _generate_qml_style_file_by_split(qml_path: Path, label_field: str) -> None:
    qml_content = _load_qml_template("style_by_split.qml", label_field=label_field)
    qml_path.parent.mkdir(parents=True, exist_ok=True)
    qml_path.write_text(qml_content, encoding="utf-8")


def _generate_qml_style_file_by_train_usage(qml_path: Path, label_field: str) -> None:
    qml_content = _load_qml_template("style_by_train_usage.qml", label_field=label_field)
    qml_path.parent.mkdir(parents=True, exist_ok=True)
    qml_path.write_text(qml_content, encoding="utf-8")
