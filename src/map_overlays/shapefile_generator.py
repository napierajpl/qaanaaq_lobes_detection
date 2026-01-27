"""
Shapefile generation for tile visualization in QGIS.
"""

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from src.map_overlays.tile_registry import TileRegistry

logger = logging.getLogger(__name__)


def generate_tile_index_shapefile(
    registry: TileRegistry,
    output_path: Path,
    label_field: str = "tile_label",
    include_all_tiles: bool = True,
) -> None:
    """
    Generate shapefile from tile registry for QGIS visualization.
    
    Args:
        registry: TileRegistry instance
        output_path: Path to output shapefile (without .shp extension)
        label_field: Field name to use for labels in QGIS
        include_all_tiles: If True, include all tiles; if False, only valid tiles
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
        
        # Build attributes
        attrs = {
            "tile_id": tile_id,
            label_field: tile_label,
            "tile_idx": tile.get("tile_idx", -1),
            "is_valid": int(tile.get("filtering", {}).get("is_valid", False)),
            "rgb_valid": int(tile.get("filtering", {}).get("rgb_valid", False)),
            "has_targets": int(tile.get("filtering", {}).get("has_targets", False)),
            "split": tile.get("split", ""),
        }
        
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
    
    # Generate QGIS style file (QML) for better visualization
    qml_path = Path(str(output_path).replace(".shp", ".qml"))
    _generate_qml_style_file(qml_path, label_field)
    logger.info(f"Generated QGIS style file: {qml_path}")
    
    # Log summary
    if "is_valid" in gdf.columns:
        valid_count = gdf["is_valid"].sum()
        logger.info(f"  Valid tiles: {valid_count}/{len(gdf)}")
    
    if "split" in gdf.columns:
        split_counts = gdf["split"].value_counts()
        logger.info(f"  Split distribution: {dict(split_counts)}")


def _generate_qml_style_file(qml_path: Path, label_field: str) -> None:
    """
    Generate QGIS style file (QML) with labels and semi-transparent styling
    to make tile overlap visible.
    
    Args:
        qml_path: Path to output QML file
        label_field: Field name to use for labels
    """
    # QML with categorized styling based on is_valid field
    qml_content = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 symbollevels="0" enableorderby="0" forceraster="0" type="categorizedSymbol" attr="is_valid">
    <categories>
      <category symbol="0" value="0" label="Not Filtered (Invalid)" render="true"/>
      <category symbol="1" value="1" label="Filtered (Valid)" render="true"/>
    </categories>
    <symbols>
      <symbol alpha="0.2" clip_to_extent="1" force_rhr="0" name="0" type="fill">
        <layer locked="0" class="SimpleFill" pass="0" enabled="1">
          <Option type="Map">
            <Option name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
            <Option name="color" value="200,150,150,255" type="QString"/>
            <Option name="joinstyle" value="bevel" type="QString"/>
            <Option name="offset" value="0,0" type="QString"/>
            <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
            <Option name="offset_unit" value="MM" type="QString"/>
            <Option name="outline_color" value="150,100,100,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="0.2" type="QString"/>
            <Option name="outline_width_unit" value="MM" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol alpha="0.4" clip_to_extent="1" force_rhr="0" name="1" type="fill">
        <layer locked="0" class="SimpleFill" pass="0" enabled="1">
          <Option type="Map">
            <Option name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
            <Option name="color" value="100,150,200,255" type="QString"/>
            <Option name="joinstyle" value="bevel" type="QString"/>
            <Option name="offset" value="0,0" type="QString"/>
            <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
            <Option name="offset_unit" value="MM" type="QString"/>
            <Option name="outline_color" value="50,100,150,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="0.3" type="QString"/>
            <Option name="outline_width_unit" value="MM" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fontItalic="0" fontKerning="1" fontLetterSpacing="0" fontStrikeout="0" fontUnderline="0" fontWordSpacing="0" fieldName="{label_field}" fontSize="8" fontSizeUnit="Point" blendMode="0" textColor="50,50,50,255" textOpacity="1" fontWeight="50" multilineHeight="1" useSubstitutions="0" fontSizeMapUnitScale="3x:0,0,0,0,0,0" previewBkgrdColor="255,255,255,255" fontFamily="Arial" namedStyle="Regular" allowHtml="0" isExpression="0">
        <text-buffer bufferSize="1" bufferSizeUnits="MM" bufferColor="255,255,255,255" bufferBlendMode="0" bufferDraw="1" bufferNoFill="0" bufferSizeMapUnitScale="3x:0,0,0,0,0,0" bufferJoinStyle="128" bufferOpacity="1"/>
      </text-style>
      <text-format wrapChar="" useMaxLineLengthForAutoWrap="1" rightDirectionSymbol=">" leftDirectionSymbol="&lt;" reverseDirectionSymbol="0" formatNumbers="0" decimals="3" placeDirectionSymbol="0" multilineAlign="3" plussign="0" addDirectionSymbol="0" autoWrapLength="0"/>
      <placement placementFlags="10" repeatDistance="0" dist="0" distUnits="MM" repeatDistanceMapUnitScale="3x:0,0,0,0,0,0" geometryGenerator="make_point(x_min($geometry), y_max($geometry))" geometryGeneratorType="PointGeometry" predefinedPositionOrder="TR,TL,BR,BL,R,L,TSR,BSR" offsetType="1" lineAnchorPercent="0.5" centroidInside="0" xOffset="5" yOffset="-5" lineAnchorType="0" rotationAngle="0" repeatDistanceUnits="MM" overrunDistance="0" priority="5" overrunDistanceMapUnitScale="3x:0,0,0,0,0,0" geometryGeneratorEnabled="1" maxCurvedCharAngleIn="25" maxCurvedCharAngleOut="-25" overrunDistanceUnits="MM" centroidWhole="0" labelOffsetMapUnitScale="3x:0,0,0,0,0,0" distMapUnitScale="3x:0,0,0,0,0,0" quadOffset="4" preserveRotation="1" layerType="PolygonGeometry" fitInPolygonOnly="0" placement="0"/>
      <rendering scaleVisibility="0" fontMinPixelSize="3" obstacle="1" upsidedownLabels="0" maxNumLabels="2000" zIndex="0" fontMaxPixelSize="10000" unplacedVisibility="0" mergeLines="0" minFeatureSize="0" limitNumLabels="0" drawLabels="1" scaleMin="0" scaleMax="0" obstacleType="1" labelPerPart="0" displayAll="0" obstacleFactor="1"/>
    </settings>
  </labeling>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>2</layerGeometryType>
</qgis>
'''
    
    qml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qml_path, 'w', encoding='utf-8') as f:
        f.write(qml_content)
