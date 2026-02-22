# Improvements Backlog

This file tracks potential improvements, ideas, and future work for the lobe detection project. Items are organized by priority and category. **Implemented items** are listed in **`docs/implemented_features_list.md`**.

**Literature alignment (2026-02):** Priorities are aligned with `docs/literature/similar_tasks_imbalance_linear_structures.md`, `docs/literature/gully_erfnet_bibliography_analysis.md`, and **`docs/literature/bibliography_deep_dive_30min.md`** (30-min sustained session 2026-02-03). Dataset size is unlikely the bottleneck; **loss design** and optionally binary targets or auxiliary heads are the main levers. The **loss order below** follows Azad et al. 2023 (25-loss survey) plus gully/road literature: best evidence for small foreground, linear structures, and imbalance.

**Priority rationale (2026-02-03):** Azad 2023 evaluated 6 losses on Synapse (small organs) and Cityscapes; **Tversky and Focal Tversky** performed best on small structures; **Jaccard** gave sharp boundaries and second-best. **Boundary + GDL** (Kervadec) and **UFL** are well supported. We keep **ACL** (Gully-ERFNet) first as direct domain analogue. Topology/connectivity losses (Skeleton Recall, clDice, Road-topology) are in Medium as next wave after region/boundary losses.

**Don’t block architecture on all 8 losses.** Do a **first wave** (items 1–4: ACL, Tversky, GDL, Boundary+GDL), then **re-evaluate**: if no clear gain, try **SE + PPM** (item 5, architecture) before the next loss wave (6–8). That way we get a signal from both loss and architecture within a few experiments.

## High Priority

### Loss + architecture (interleaved)

**First wave (1–4):** Loss only. **Then** consider architecture (5) before the next loss wave (6–8).

1. **ACL-style loss (Dice + Focal with λ)** ⭐ (Highest Priority)
   - **Why**: Gully-ERFNet (Li et al., IJDE 2025) uses Adaptive Correction Loss = λ·Dice + (1−λ)·Focal for gully extraction (linear structures, severe imbalance, label noise). Direct remote-sensing analogue; we already have Dice and Focal in codebase.
   - **What**: Implement combined Dice + Focal with a tunable or scheduled λ; compare to current combined (IoU + weighted Smooth L1) on val MAE and improvement over baseline. Can use **binary target** (thresholded proximity) for this experiment if desired — binary is a valid option (see literature doc § "Why proximity maps instead of binary").
   - **Status**: Not implemented
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §4 Gully-ERFNet

2. **Tversky loss / Focal Tversky loss**
   - **Why**: Azad et al. 2023 (arXiv:2312.05391): **Tversky and Focal Tversky performed best** on Synapse (small organs: pancreas, gallbladder); Dice and Focal showed constant/plateau behavior. Tversky generalizes Dice with α, β for FP/FN trade-off; Focal Tversky adds hard-example focus. Direct fit for small foreground and imbalance.
   - **What**: Add Tversky loss (and optionally Focal Tversky, γ ∈ [1,3]); compare to ACL and current Combined on val MAE. Use binary (thresholded proximity) or regression head as now; tune α, β if needed.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/bibliography_deep_dive_30min.md` §9 Round 14; Azad 2023 §2.2.6–2.2.7

3. **Generalized Dice loss**
   - **Why**: ResUNet-a (ISPRS 2020) uses a Generalized Dice variant for highly imbalanced remote sensing; Sudre et al. 2017: class weight \(w_\ell = 1/(\sum_i r_i^\ell)^2\) rebalances by inverse volume so small (lobe) class contributes fairly.
   - **What**: Implement GDL (Sudre formula); compare to our Combined loss. Option: use GDL inside ACL instead of standard Dice. Implementation: gravitino/generalized_dice_loss or pytorch-3dunet.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §1; `docs/literature/bibliography_deep_dive_30min.md` §1.1

4. **Boundary loss (Kervadec) + GDL or Dice**
   - **Why**: Kervadec et al. (MedIA): contour-distance integral; +8% Dice when combined with GDL. Boundary-Aware U-Net (glacier): 0% → 35.9% IoU with boundary term. Direct fit for boundary/localisation issues.
   - **What**: Use LIVIAETS/boundary-loss (PyTorch); combine with GDL or Dice. Check if combinable with proximity regression (e.g. auxiliary binary head).
   - **Status**: Not implemented
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §2; `docs/literature/bibliography_deep_dive_30min.md` §1.2

5. **SE + Pyramid Pooling Module (PPM)** (architecture — try after first loss wave) ✅ **IMPLEMENTED**
   - **Why**: Gully-ERFNet adds SE (channel attention) and PPM for multi-scale context; same linear/thin/imbalance setting. Lightweight; proven for gully extraction. Don’t wait for all 8 loss experiments — re-evaluate after items 1–4; if loss gains are small, try SE+PPM next.
   - **What**: Add SE blocks and PPM between encoder and decoder (e.g. after SatlasPretrain encoder); compare val MAE vs baseline. Yu et al. (IEEE JSTARS 2018) is the RS-oriented PPM source.
   - **Status**: ✅ Implemented. Config: `model.use_se`, `model.use_ppm` (flat); optional `ppm_bins`, `se_reduction`. Ablation: set true/false per run. See `docs/project_features.md` §1 and §14.
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §4 Gully-ERFNet; `docs/literature/gully_erfnet_bibliography_analysis.md`

6. **Jaccard (IoU) loss**
   - **Why**: Azad 2023: Jaccard loss gave **second-best** on Synapse (small organs), sharp boundaries, overlap-based; stable training vs Dice/Focal. Good for imbalanced class distributions.
   - **What**: Add IoU/Jaccard loss (relaxed, differentiable); compare to Combined and ACL. Rahman et al.; many PyTorch impls; also in YilmazKadir/Segmentation_Losses.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/bibliography_deep_dive_30min.md` §9 Round 14; Azad 2023 §2.2.4

7. **Unified Focal Loss (UFL)**
   - **Why**: Yeung et al. (Medical Image Analysis 2022) show UFL outperforms CE, Focal, Dice, Tversky on extreme imbalance; Focal alone can worsen vs CE. Our Focal runs led to "predict constant"; UFL may avoid that.
   - **What**: Add UFL (e.g. tayden/unified-focal-loss-pytorch); run with UFL instead of Focal or Combined; for proximity either binarize for UFL or use UFL as auxiliary head.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §3

8. **Combo loss (Dice + WCE) or Log-cosh Dice**
   - **Why**: Azad 2023: Combo and Focal loss showed **most consistent penalization** for small objects (Figure 3); Log-cosh Dice (Jadon 2020) smooths Dice, better precision/recall balance. Low-risk baselines.
   - **What**: Try Combo (α·WCE + (1−α)·Dice) or Log-cosh Dice as alternative to current Combined; compare val MAE. Jadon GitHub: shruti-jadon/Semantic-Segmentation-Loss-Functions; Azad: YilmazKadir/Segmentation_Losses.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/bibliography_deep_dive_30min.md` §9; Jadon 2020; Azad 2023 §2.4.1, §2.2.2

9. **Increase Encouragement Weight** (quick test)
   - **Why**: Current weight=10.0 may be insufficient to break local minimum.
   - **What**: Try 50.0 or 100.0 for stronger penalty on under-predicting lobe pixels.
   - **Status**: Not tested
   - **Reference**: Daily diary 2026-01-21

### Architecture

- **Pretrained Encoder (SatlasPretrain)** — Implemented. See `docs/implemented_features_list.md`. Option A (segmentation-models-pytorch) and Option C (TorchGeo) not implemented.

## Medium Priority

### Loss (topology / connectivity) — after region and boundary losses

Try after the High Priority loss experiments above; same linear/thin/imbalance setting.

- **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024): Thin tubular (vessels, roads, cracks); multi-class; >90% less compute. GitHub: MIC-DKFZ/Skeleton-Recall. **Priority:** first topology loss to try.
- **clDice (centerline Dice)** (Shit et al., CVPR 2021): Topology-preserving for tubular/linear; soft-clDice differentiable. GitHub: jocpae/clDice. Reduces broken/scattered predictions.
- **Road-topology loss** (MDPI Remote Sensing 13:2080): Penalizes gaps and spurious segments; +11.98% IoU on road detection. Same linear/thin/imbalance as lobes.
- **GapLoss / NeighborLoss** (road RS): GapLoss for boundary/continuity; NeighborLoss for spatial correlation. GitHub: Dabao55/GapLoss, chinaericy/neighborloss.
- **Skea-Topo** (IJCAI 2024): Skeleton-aware + boundary rectified term; +7 pt VI. GitHub: clovermini/skea_topo. Optional after Skeleton Recall / clDice.

Reference: `docs/literature/bibliography_deep_dive_30min.md` §1.4, §1.5, §9.

### Training Strategy

11. **Warm-up Training**
   - **Why**: Escape local minimum before fine-tuning
   - **What**: High LR (0.1) for 5-10 epochs, then reduce to normal LR
   - **Status**: Not implemented
   - **Reference**: Identified in daily diary 2026-01-21

12. **Learning Rate Adjustments**
   - **Why**: LR=0.01 may be too high, causing early plateau
   - **What**: Experiment with lower initial LR (0.001-0.005) or better scheduling
   - **Status**: Partially addressed (scheduler exists but doesn't trigger)

### Data

13. **Stone-stripe / slope-aligned texture hint channel(s)**
   - **Why**: Lobes occur where stones form stripes; non-lobe areas are finer-grained. Stripes are directional — perpendicular to lobe front and **follow slope direction**. A mechanism that detects “stripes following slope” and exposes it (e.g. as an input channel or visualization) could give the CNN a strong hint for lobe vs non-lobe.
   - **What**: Detect stripes that follow slope direction and surface them. Options: (1) Structure-tensor from RGB → coherence (stripiness) + dominant orientation; aspect from DEM (slope direction); one or more channels: e.g. “slope–texture alignment” (high where local texture direction matches aspect). (2) Precomputed raster(s) tiled like segmentation, or on-the-fly in the dataloader from RGB + DEM. (3) Optional: visualization/QC layer (e.g. slope-aligned stripe strength) to inspect where the detector fires.
   - **Status**: Not implemented
   - **Reference**: User request (stripes following slope; mechanism to detect and show). See also `docs/plan_synthetic_parenthesis_and_multiscale.md` §3 (boundary/segmentation as input hints).

14. **Class-Balanced Sampling**
   - **Why**: Most batches are mostly background, model sees few lobe examples
   - **What**: Oversample tiles with high lobe density during training
   - **Status**: Not implemented



16. **Full Mosaic / GeoTIFF Export** (prediction tile visualization)
   - **Why**: Visual sanity check of model outputs across the full AOI (not just scalar metrics). Per-tile representative visualization is already implemented (see `docs/implemented_features_list.md`).
   - **What**:
     - Save per-tile predictions as GeoTIFF tiles (aligned to target/proximity tiles)
     - Stitch tiles back into a single georeferenced raster (mosaic) for the AOI
     - Visual compare prediction raster vs proximity map in QGIS (difference/overlay)
   - **Open questions**:
     - How expensive is writing all tiles + mosaicking for production? (disk + time)
     - Do we export full float raster, or a compressed/quantized product for visualization?
   - **Status**: Not implemented



### Research / Literature

- **Literature search: similar tasks (linear/elongated, imbalance)** — Done. See `docs/implemented_features_list.md` and `docs/literature/similar_tasks_imbalance_linear_structures.md`.

### Post-processing (literature-backed)

19. **Post-processing for lobe maps (small-object/hole removal)**
   - **Why**: Gully-ERFNet uses remove_small_objects and remove_small_holes (e.g. 1000 px) plus road-buffer exclusion; improves precision/recall. We could apply similar cleanup to thresholded lobe predictions for final maps.
   - **What**: After inference, optionally apply morphology (remove_small_objects, remove_small_holes) to binarized prediction; optionally mask known non-lobe features (roads, etc.) if vector data available.
   - **Status**: Not implemented
   - **Reference**: `docs/literature/similar_tasks_imbalance_linear_structures.md` §4 Gully-ERFNet

### From Gully-ERFNet bibliography analysis (2026-02)

Source: `docs/literature/gully_erfnet_bibliography_analysis.md`. Loss ideas are consolidated into High/Medium priorities above.

- **Log-cosh Dice loss** (Jadon 2020): Smooth Dice variant; try as alternative to standard Dice in ACL or combined loss. GitHub: shruti-jadon/Semantic-Segmentation-Loss-Functions.
- **Spatial consistency / NeighborLoss** (Xu 2023): Loss term that encourages neighbor agreement for linear structures (roads); could reduce scattered lobe predictions. Explore spatial-consistency term alongside current loss.
- **Slope-of-slope or curvature** (Chen 2024): When DEM resolution allows, add slope-of-slope (abruptness) or curvature as extra input channel; improved gully mapping in their setup. Optional after DEM quality check.
- **DEM quality check and ablation** (Liu 2022, Gafurov 2020): DEM vertical accuracy can matter more than resolution; consider ablating DEM/slope or downweighting if our DEM is coarse or noisy.
- **Non-local / self-attention for linear structures** (Zhu 2024): Asymmetric Non-Local LinkNet for gully; long-range context along linear features. Consider non-local or self-attention in decoder for along-lobe context (lower priority than loss).
- **Foreground-driven fusion** (Shen 2024): Foreground (gully) emphasis in fusion; we already use lobe weighting — could formalize as explicit foreground-driven loss or fusion if we add more modalities later.
- **clDice (centerline Dice)** (Shit et al., CVPR 2021): Topology-preserving loss for **tubular/linear** structures (vessels, roads); soft-clDice is differentiable. Could reduce broken/scattered lobe predictions. GitHub: jocpae/clDice. See `docs/literature/bibliography_deep_dive_30min.md` §1.4.
- **GapLoss / NeighborLoss** (road segmentation RS): GapLoss for boundary; NeighborLoss for spatial consistency. GitHub: Dabao55/GapLoss, chinaericy/neighborloss. See bibliography_deep_dive_30min.md §1.5.

### From 30-min sustained bibliography session (2026-02-03)

Source: `docs/literature/bibliography_deep_dive_30min.md` §9. Skeleton Recall, Road-topology, Skea-Topo, GapLoss/NeighborLoss → Medium "Loss (topology/connectivity)". Azad 2023 top 5 → High Priority loss order (items 1–7).

- **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024): For thin tubular structures (vessels, roads, cracks); **multi-class**; >90% less compute than other topology losses (CPU skeleton + GPU soft recall). Strong candidate for lobe connectivity. GitHub: MIC-DKFZ/Skeleton-Recall.
- **Road-topology loss** (MDPI Remote Sensing 13:2080): Ordinal regression + topology loss penalizing gaps and spurious segments; +11.98% IoU on road detection. Same linear/thin/imbalance as lobes; consider as auxiliary or alternative to GapLoss.
- **Skea-Topo** (IJCAI 2024, arXiv:2404.18539): Skeleton-aware weighted loss + boundary rectified term (BoRT); +7 pt VI on boundary segmentation. GitHub: clovermini/skea_topo. Optional after Skeleton Recall / clDice.
- **Azad et al. 2023 loss survey** (arXiv:2312.05391): 25 loss functions, taxonomy, PyTorch implementations. Use as systematic try-list: pull full list, rank top 5 for “small foreground, linear, imbalance”; GitHub: YilmazKadir/Segmentation_Losses.

## Low Priority / Future Work

20. **Strip Convolutions for Linear Features**
   - **Why**: Lobes are linear/elongated features; strip convolutions excel at high aspect ratio objects.
   - **What**: Incorporate strip convolution concepts from Strip R-CNN (82.75% mAP on DOTA-v1.0).
   - **Complexity**: High — requires architecture modifications.
   - **Reference**: Strip R-CNN paper - "Large Strip Convolution for Remote Sensing Object Detection"
   - **Status**: Research phase

21. **Two-Stage Training**
   - **Why**: Separate detection from regression; literature (ResUNet-a) uses multi-task (boundary → distance → mask).
   - **What**: Stage 1: Binary segmentation (lobe vs background) with Dice/Focal or ACL; Stage 2: Fine-tune for proximity on lobe pixels or add auxiliary proximity head.
   - **Status**: Not implemented
   - **Complexity**: High — requires pipeline changes

22. **Attention Mechanisms**
   - **Why**: Focus model attention on important regions; SE+PPM is High Priority item 5.
   - **What**: Other attention (spatial, etc.) if SE+PPM is insufficient.
   - **Status**: Not implemented

23. **Deep Supervision**
   - **Why**: Add auxiliary losses at intermediate layers.
   - **What**: Compute loss at multiple decoder levels.
   - **Status**: Not implemented

24. **Ensemble Methods**
   - **Why**: Combine multiple models for better predictions.
   - **What**: Train multiple models, average predictions.
   - **Status**: Not implemented

Implemented items (dropout, gradient clipping, LR scheduling, early stopping, per-tile baseline, training visualization, MLflow, SatlasPretrain U-Net) are listed in **`docs/implemented_features_list.md`**.

## Scored & ranked (Performance 0–5 × Effort 0–5, sort by sum)

**Scoring:** Performance = chance of meaningful model improvement (5 = high). Effort = ease of implementation (5 = easy / few lines). **Higher sum = do first.** Ties broken by performance.

| Rank | Sum | Perf | Effort | Improvement |
|------|-----|------|--------|-------------|
| 1 | 10 | 5 | 5 | **ACL-style loss (Dice + Focal with λ)** — We have both losses; combine with λ. Direct gully analogue. |
| 1 | 10 | 5 | 5 | **Jaccard (IoU) loss** — Standard differentiable IoU; Azad second-best on small organs; drop-in. |
| 3 | 9 | 5 | 4 | **Tversky loss / Focal Tversky loss** — Azad best on small structures; add loss, tune α,β. |
| 3 | 9 | 5 | 4 | **Generalized Dice loss** — Sudre formula; class weights; well-established for imbalance. |
| 3 | 9 | 4 | 5 | **Combo loss (Dice + WCE) or Log-cosh Dice** — Low-risk baseline; we have Dice, WCE is standard. |
| 6 | 8 | 5 | 3 | **Boundary loss (Kervadec) + GDL or Dice** — Strong papers (+8% Dice); integrate LIVIAETS/boundary-loss. |
| 6 | 8 | 4 | 4 | **Unified Focal Loss (UFL)** — Good for extreme imbalance; external PyTorch impl. |
| 8 | 7 | 4 | 3 | **SE + Pyramid Pooling Module (PPM)** — Architecture; add SE + PPM to decoder. |
| 8 | 7 | 3 | 4 | **Learning Rate Adjustments** — Lower LR or better schedule; config + scheduler. |
| 8 | 7 | 3 | 4 | **Data Augmentation** — Flips, rotations, photometric; care with DEM. |
| 8 | 7 | 3 | 4 | **Class-Balanced Sampling** — Oversample lobe-rich tiles; sampler change. |
| 8 | 7 | 4 | 3 | **Stone-stripe / slope-aligned texture channel** — Detect stripes following slope (structure tensor + aspect); add as input channel or QC layer. |
| 8 | 7 | 4 | 3 | **Background tiles + augment lobe tiles only** — 4× background tiles, 4× augmentation for lobe tiles only; slope becomes useful. |
| 8 | 7 | 3 | 4 | **Post-processing (small-object/hole removal)** — Morphology at inference; no training change. |
| 8 | 7 | 2 | 5 | **Increase Encouragement Weight** — Change one constant (e.g. 50 or 100). |
| 14 | 7 | 4 | 3 | **GapLoss / NeighborLoss** — Road RS; impls exist; integrate. |
| 15 | 6 | 4 | 2 | **Skeleton Recall Loss** — Topology for tubular; integrate MIC-DKFZ/Skeleton-Recall. |
| 15 | 6 | 4 | 2 | **clDice (centerline Dice)** — Topology-preserving; integrate jocpae/clDice. |
| 15 | 6 | 2 | 4 | **Warm-up Training** — High LR for few epochs then reduce. |
| 18 | 5 | 3 | 2 | **Road-topology loss** — Ordinal + topology; more involved. |
| 18 | 5 | 3 | 2 | **Skea-Topo** — Skeleton + boundary term; clovermini/skea_topo. |
| 18 | 5 | 4 | 1 | **Two-Stage Training** — Pipeline change; Stage 1 binary, Stage 2 proximity. |
| 18 | 5 | 3 | 2 | **Attention Mechanisms** — Beyond SE+PPM if needed. |
| 22 | 4 | 2 | 2 | **Deep Supervision** — Auxiliary losses at decoder levels. |
| 22 | 4 | 3 | 1 | **Strip Convolutions** — Architecture change; strip convs. |
| 22 | 4 | 2 | 2 | **Ensemble Methods** — Train multiple models, average predictions. |
| 22 | 4 | 2 | 2 | **Experiment with Larger Tiles** — Retiling + batch size. |
| 26 | 3 | 0 | 3 | **Prediction Tile Visualization + Mosaic Raster Export** — No model performance change; tooling. |

**Done (not scored):** See `docs/implemented_features_list.md` (Pretrained Encoder, Literature search, and other completed items).

## Notes

- **Priority rationale (2026-02-03):** Loss order follows Azad et al. 2023 (Synapse small organs: Tversky/Focal Tversky best, Jaccard second) plus gully/glacier literature (ACL, GDL, Boundary). Topology/connectivity losses (Skeleton Recall, clDice, Road-topology) are Medium, to try after region/boundary losses.
- Literature: `docs/literature/similar_tasks_imbalance_linear_structures.md`, `docs/literature/gully_erfnet_bibliography_analysis.md`, `docs/literature/bibliography_deep_dive_30min.md`.
- Dataset size is unlikely the bottleneck; **loss design** and optionally binary targets or auxiliary heads are the main levers.
- When an item is implemented, add it to `docs/implemented_features_list.md` and remove or shorten its entry here.
- **Scored & ranked** above: use **Perf + Effort** sum to pick quick wins (e.g. sum ≥ 9 first: ACL, Jaccard, Tversky, GDL, Combo).
