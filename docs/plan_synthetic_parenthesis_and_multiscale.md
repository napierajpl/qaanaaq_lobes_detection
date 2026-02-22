# Plan: Synthetic parenthesis from full raster + boundary vectors + segmentation layer

## Goals

1. **Image boundary vectors (write once, reuse)**
   - Find boundaries of the image (distinct polygons: valid-data regions, e.g. non-white / non-nodata).
   - Define them as **vectors** (GeoJSON or shapefile), write once.
   - Use for: constrain parenthesis placement to **inside** those boundaries; reusable for other purposes (e.g. masking, QC).
   - Placement can use **geo coordinates or pixels** (we support both via rasterizing the vector to a mask when needed).

2. **Synthetic parenthesis pipeline (new)**
   - Use full raster `qaanaaq_rgb_0_2m.tif`; embed parenthesis in **random places inside boundary polygons only** (and avoid white background).
   - Produce the **same target raster for learning** (0 outside, 20 inside parenthesis).
   - Then create **tile sets** (256, 512, or other sizes) from the result.

3. **Segmentation as a separate layer (independent process)**
   - **Separate operation** that creates a **separate raster layer** (e.g. segment IDs or boundary map).
   - **Independent** of synthetic vs normal dataset: can be run on **any** raster (normal imagery or imagery with parenthesis).
   - Used as **hint for the CNN** (e.g. extra channel or auxiliary input to help find object boundaries).
   - Does **not** define the training target; the target for learning remains the same (e.g. 0/20 parenthesis mask or lobe proximity).

---

## 1. Current vs desired flow

### Current (existing script)

- **Input:** Existing 256/512 **tiles** (from dev or train: `filtered_tiles.json` + `features/`).
- **Per tile:** Load 5-band tile → draw parenthesis at random positions on the tile → burn black into RGB, set target 20 inside / 0 outside → write tile.
- **Output:** One tile size per run. No boundary constraint; no segmentation layer.

### Desired

- **Boundary extraction (once):** From the reference raster, compute valid-data mask (non-white, optional non-nodata) → vectorize to polygons → write vector file. Reusable for placement and other uses.
- **Synthetic pipeline:** Load RGB + boundary (vector or rasterized mask). Place **2000 parenthesis**, **300 px height** each, at random positions **inside boundaries** and avoiding white. Build **target raster** (0/20). Stack RGB (with shapes) + DEM + Slope → tile 256 and 512 → write `features/`, `targets/`, `filtered_tiles.json` per tile size. **Target for learning = same** (parenthesis mask 0/20).
- **Segmentation (separate):** Standalone script: input = any raster; output = **segmentation layer** raster (same grid). Can be run on normal dataset or on dataset with parenthesis. Result is an extra layer (e.g. 6th band or separate file) giving boundary/segment hints to the CNN.

---

## 2. Technical design

### 2.1 Boundary extraction

- **Input:** Reference raster (e.g. `qaanaaq_rgb_0_2m.tif`). **Valid** = not white (all RGB >= 250) and optionally not nodata.
- **Process:** Build binary mask (1 = valid, 0 = invalid) → `rasterio.features.shapes` with transform → polygons in geo coordinates → GeoDataFrame → write **GeoJSON** (and optionally .shp). One or more distinct polygons (e.g. main landmass, islands).
- **Output:** Single vector file (e.g. `data/processed/vector/imagery_valid_boundaries.geojson`). Reusable for: parenthesis placement, masking, QC, any future “inside image” logic.
- **Use in synthetic pipeline:** Rasterize vector to mask (same shape as RGB) or test point-in-polygon (pixel → geo) when sampling placement positions.

### 2.2 Full-raster parenthesis placement

- **Source:** Full RGB + boundary mask (from vector or precomputed). **Count:** 2000 shapes, **height:** 300 px each.
- **Sample positions:** Random (row, col) only where: (1) boundary mask is 1, (2) local patch is not too white (e.g. fraction white in footprint < threshold). Use same white definition as dataloader (all bands >= 250).
- **Draw:** Reuse existing parenthesis rendering (PIL, rotate, scale). Burn black into RGB; build target raster 0/20. **Target for learning = this raster** (unchanged).
- **DEM/Slope:** Resample to match RGB (as in current pipeline), stack RGB (with shapes) + DEM + Slope for 5-band features.
- **Memory:** Raster ~35000×35000; 96 GB RAM. Float32 5-band ≈ 24 GB; full load is feasible. Implement full in-memory first; optional chunked fallback if needed.

### 2.3 Segmentation as separate layer

- **Role:** Precomputed **segmentation layer** as **input hint** to the CNN (boundary/object hints), not as the training target. Literature: boundary-conditioned backbones and joint boundary+segmentation tasks improve segmentation (e.g. +0.5–3% IoU, better boundary F-scores). Whole objects (e.g. parenthesis) often form single segments; surroundings more fragmented.
- **Process:** **Standalone script.** Input: any raster (path). Output: one raster (same grid) with segment IDs or boundary map. Method: open-source OBIA-style (e.g. `skimage.segmentation.felzenszwalb` or SLIC). **Scale:** Suggest **2 scales** (e.g. fine + coarse); output can be one layer per scale or combined (e.g. segment ID band).
- **Limited to boundary:** By default (or with `-b`), segmentation is limited to the research boundary: pixels outside are written as nodata. Same boundary vector used for synthetic placement.
- **Use:** Run on normal imagery or on imagery with parenthesis. Result is a **separate layer** (separate file or optional 6th band); training target is unchanged.

### 2.4 Limiting training to research boundary

- **Tile filter:** `filter_tiles_by_boundary.py` takes `filtered_tiles.json` and the boundary vector and outputs a new tile list containing only tiles that intersect the boundary (using registry or feature-dir for tile bounds). Use this list as `filtered_tiles` in config so training, validation, and tuning run only on tiles inside the research boundary.

### 2.4 Tiling and outputs

- **Features:** 5-band (RGB with shapes + DEM + Slope). **Target:** 1-band (0/20 parenthesis). Same layout as now.
- **Tiling:** `Tiler` for 256 and 512, same overlap (0.3). Write `synthetic_parenthesis_256/` and `synthetic_parenthesis_512/` with `features/`, `targets/`, `filtered_tiles.json`.

---

## 3. Research note: segmentation combined with CNN

- Using **precomputed segmentation or boundary maps as input** to CNNs is well established: (1) **Boundary-conditioned backbone:** semantic boundary detection as auxiliary task or extra input improves segmentation (e.g. SBCB, boundary-aware CNN). (2) **Joint multi-task:** boundary detection and segmentation share features; boundaries help around edges. (3) **Auxiliary channel:** an extra input channel with segment IDs or boundary strength gives the network explicit boundary/object hints. So our “segmentation as separate layer” fits: we produce a **hint layer** that the model can use; the **target** stays the lobe/parenthesis mask.

---

## 4. Implementation summary

1. **Boundary extraction:** Script that reads raster → valid mask (non-white, optional non-nodata) → vectorize → write GeoJSON. Optional: rasterize vector to mask for downstream use.
2. **Synthetic from raster:** Load RGB + boundaries (vector → rasterize to mask). Sample 2000 positions inside mask and not white; place 300 px parenthesis; build target 0/20; stack with DEM/slope; tile 256/512; write filtered_tiles.json. Same target raster for learning.
3. **Segmentation layer:** Standalone script: input raster, output segmentation raster (same grid). Method: skimage (e.g. felzenszwalb); suggest 2 scale parameters. Independent of dataset type.
4. **Docs:** Update synthetic_parenthesis_dataset.md and gully_erfnet_multiscale_segmentation.md.
