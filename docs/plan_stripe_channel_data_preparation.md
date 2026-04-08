# Plan: Stripe / Slope-Aligned Linear Channel (Data Preparation)

## Goal

Add a **new data preparation process** (same flow as segmentation): **process whole area first**, then **cut into tiles**. The new **channel** should provide information about **stripes / linear shapes that follow slope**, using DEM + slope + RGB. For development, use **representative 512×512 tiles**.

---

## Two-stage process: stripes → slope alignment

What matters for us is **not only** presence of stripes **but** whether stripes **follow the slope**. So the pipeline is explicitly two-stage:

1. **Stage 1 — Stripe presence (and direction):** Detect / measure linear structure in the image (RGB).
   - Output: *Is there a dominant stripe direction here?* (e.g. **coherence** from structure tensor, or Gabor strength) and *which direction?* (**stripe_angle**).
2. **Stage 2 — Slope alignment:** Compare that stripe direction to terrain.
   - Use **aspect** (direction of steepest descent from DEM).
   - Output: *Do the stripes align with the slope?* (e.g. **slope_alignment** in [0, 1]: 1 = aligned, 0 = perpendicular).

The **relevant signal** for lobe/stripe detection is the **combination**: high only where (1) there are stripes **and** (2) they follow the slope. So we expose either two channels (Stage 1 + Stage 2) so the model can combine them, or one channel (e.g. coherence × slope_alignment) that is strong only when both hold.

---

## 1. Literature Summary

### 1.1 Problem domain

- **Stone stripes** (periglacial): linear accumulations aligned with steepest slope; spacing and orientation are slope-driven. Detection benefits from DEM (aspect), slope, and imagery.
- **Linear feature detection** in remote sensing: lineaments, terrace ridges, gullies, sorted patterned ground — often use **directional** and **slope-aligned** cues.

### 1.2 Relevant methods

| Approach | Source / use | Relevance |
|----------|--------------|-----------|
| **Structure tensor** | Gradient structure tensor (J₁₁, J₁₂, J₂₂) → eigenvalues → **coherence** (anisotropy) and **orientation** (stripe direction). | High: we already use this in `src/preprocessing/texture_hints.py`: coherence + stripe_angle from RGB. |
| **Slope alignment** | Compare local texture/stripe direction to **aspect** (direction of steepest descent from DEM). | High: `slope_alignment(stripe_angle, aspect)` in texture_hints; 1 when aligned, 0 when perpendicular. |
| **DTM / DEM-based linear features** | Principal curvature, ridgelines, linear valleys; contour-directional detection from DEM + imagery. | Medium: could add curvature or aspect-derived bands later. |
| **UAV + DEM for patterned ground** | Automated mapping of relict patterned ground using DEM and drainage (e.g. D8). | Medium: confirms DEM + imagery is standard. |
| **Gabor filters** | Orientation- and scale-specific linear/edge detection. | Alternative: more parameters; structure tensor is simpler and already implemented. |

**Conclusion:** The existing **structure tensor + aspect + slope alignment** pipeline in `texture_hints.py` is well aligned with literature. The gap is that it is **not** run as a **whole-area raster** step; we need to produce a **stripe channel raster** (same grid as RGB/DEM), then tile it.

---

## 2. Current Codebase

- **Segmentation flow:**
  `create_segmentation_layer.py` → whole raster (block processing) → output GeoTIFF → **tile** via `create_tiles.py` → dataloader loads 6th channel from `segmentation_dir` by `tile_id`.
- **Features:** VRT = RGB + DEM + Slope (5 bands); tiles in `features/`; optional 6th channel from `segmentation_dir`.
- **Texture hints:** `src/preprocessing/texture_hints.py` has:
  - `structure_tensor_coherence_and_orientation(rgb)` → coherence (H,W), stripe_angle (H,W)
  - `aspect_from_dem(dem)` → aspect (H,W)
  - `slope_alignment(stripe_angle, aspect)` → (H,W) in [0,1]
  - `compute_texture_hint_channels(rgb, dem)` → (2, H, W): [coherence, slope_alignment]

These are **not** currently used in the dataloader; they are good candidates for the new channel(s).

---

## 3. Proposed Solution

### 3.1 Channel content

- **Implemented: single band** — `stripe_strength = coherence * slope_alignment` (high only where stripes exist **and** follow slope). Values in [0, 1]; no standardization.

### 3.2 Pipeline (implemented)

1. **Inputs:** RGB raster, DEM raster (same extent and resolution as RGB; resampling already in `prepare_training_steps`).
2. **Script:** `scripts/create_slope_stripes_channel.py` — reads RGB + DEM; block processing; outputs 1 band float32 [0, 1] (e.g. `data/processed/raster/slope_stripes_channel.tif`).
3. **Tiling:** `create_tiles.py` on slope-stripes raster → `data/processed/tiles/<tile_dir>/slope_stripes_channel/` (same tile size/overlap as features, `--no-organize`).
4. **Training:** `data.use_slope_stripes_channel: true` and `paths.*.slope_stripes_channel_dir`. Dataloader loads by `tile_id`; model `in_channels` = 5 + (1 if seg) + (1 if slope_stripes). Independent of segmentation.

### 3.3 Integration (implemented)

- **prepare_training_steps:** After resampling DEM/slope, “Generating slope-stripes channel (RGB + DEM)”; after tiling targets, “Creating tiles for slope-stripes channel”. Prod and dev both include these. **Naming:** `slope_stripes_channel`.

### 3.4 Development

- Dev: full dev pipeline (cropped 1024×1024 → slope-stripes → tile). Representative tiles from config (`visualization.representative_tile_ids*`). Representative channel figures include SlopeStripes panel when channel is enabled.

---

## 4. Resolved choices (implemented)

| Question | Decision |
|----------|----------|
| Bands | 1 band (coherence × slope_alignment) |
| With segmentation | Independent; can use both (e.g. 8 channels) |
| Normalization | Keep [0, 1], no standardization |
| Dev representative tiles | Use representative tile IDs from config; full dev pipeline generates slope-stripes then tiles |
| Naming | `slope_stripes_channel` |

**Implementation:** Script `create_slope_stripes_channel.py`, pipeline steps in `prepare_training_steps`, config `use_slope_stripes_channel` / `slope_stripes_channel_dir`, dataloader and visualization include SlopeStripes when enabled.

**Chosen structure-tensor params (from visual comparison):** `sigma_smooth=1.5`, `sigma_structure=3.0`. Use these in `create_slope_stripes_channel.py` (e.g. `--sigma-smooth 1.5 --sigma-structure 3.0`) for production. Alternative: Gabor-based method (see `run_gabor_slope_stripes_parameter_sweep.py`).
