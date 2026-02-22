# Multiscale segmentation in Gully-ERFNet (and relevance to our project)

## What “multiscale segmentation” is in the paper

In the Gully-ERFNet paper, **multiscale segmentation** is **not** a layer inside the neural network. It is used in **data preprocessing and sample generation** (Section 2.3.1).

- **Where:** After PCA on the imagery, the authors use **PIE software** (an OBIA — Object-Based Image Analysis — tool) to perform **multiscale image segmentation** (cited: Kotaridis and Lazaridou 2021).
- **What it does:** The method **divides the image into homogeneous segments** at one or more scale parameters. Those segments are then used to **select gully regions and generate preliminary gully masks** (semi-automated labels).
- **Purpose:** To **reduce manual annotation** and get a first-pass label set that is later cross-validated with land survey and aerial imagery. So it is a **label-creation / data-prep** step, not a trainable module.

So in the paper:

- **Multiscale segmentation** = OBIA multiscale segmentation in **data preprocessing** → used to create/refine **training labels**.
- **Multiscale inside the model** = **Pyramid Pooling Module (PPM)** in the network, which pools encoder features at several grid sizes and concatenates them to capture **multiscale context** in feature space.

## How multiscale is used inside Gully-ERFNet (the model)

Inside the network, multiscale context is handled by the **PPM** (and channel emphasis by **SE**):

- **PPM:** Applies pooling at **multiple scales** (e.g. different grid sizes) to the encoder output, then concatenates the upsampled results and fuses them. This gives the decoder both fine and coarse context (good for varying gully sizes and shapes).
- **SE:** Squeeze-and-Excitation reweights channels; it does not perform spatial multiscale segmentation.

So in the **model**, “multiscale” = **PPM** (multi-scale **feature** aggregation), not OBIA segmentation.

## Do we need this “method” as a new layer?

### 1. If you mean “multiscale segmentation” in the paper sense (OBIA)

- That is **not a layer**; it is a **data preparation / labeling** pipeline.
- **Possible use in our project:** We could add an optional step that runs OBIA multiscale segmentation on our imagery (e.g. with a library or external tool) to get preliminary segment masks, then use them to assist or refine lobe labels (e.g. segment boundaries, consistency checks). That would be a **data prep / pipeline** change, not a new NN layer.
- **Recommendation:** Consider only if we want to invest in semi-automated label creation or label refinement; it does not replace or add a layer in the current model.

### 2. If you mean “multiscale” inside the model (like PPM)

- We **already** use a **Pyramid Pooling Module** and **SE** in our SatlasPretrain U-Net (`src/models/se_ppm.py`, integrated in `satlaspretrain_unet.py`), inspired by the same paper.
- **Recommendation:** No new layer needed for multiscale context; we already have PPM (and SE). We can tune `use_ppm` / `use_se` and `ppm_bins` / `se_reduction` for ablation and performance.

## Our implementation: segmentation as separate layer

We added a **standalone segmentation layer** (PIE-style OBIA, using open-source Felzenszwalb):

- **Script:** `scripts/create_segmentation_layer.py`. Input: any raster; output: same-grid raster with segment IDs (1 or 2 bands for two scales).
- **Role:** **Separate layer** only — gives the CNN boundary/object hints as an extra input (e.g. 6th band or auxiliary file). It does **not** define or change the training target. Can be run on normal imagery or on imagery with parenthesis.
- **Reference:** Kotaridis & Lazaridou 2021; Gully-ERFNet Section 2.3.1 (OBIA in data prep). We use the same idea (homogeneous segments) as an optional **hint** input, not for label creation.

## Summary

| Term in paper              | Meaning                         | In our project                          |
|----------------------------|----------------------------------|-----------------------------------------|
| Multiscale segmentation    | OBIA for label/sample creation   | Optional **separate layer** (`create_segmentation_layer.py`) as CNN hint |
| Multiscale in the model    | PPM (multi-scale feature pooling)| Implemented (`se_ppm.py`, PPM + SE)      |

So: **multiscale segmentation** in the paper = label-creation preprocessing (OBIA). We use OBIA-style segmentation as a **separate raster layer** (hint for learning), not to create labels. **Multiscale in the model** = PPM, which we already have.
