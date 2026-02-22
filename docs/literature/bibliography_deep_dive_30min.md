# 30-minute bibliography deep dive: loss functions and linear-structure segmentation

Follow-up to the Gully-ERFNet bibliography analysis and `similar_tasks_imbalance_linear_structures.md`. This document summarizes **deeper findings** from following bibliographies and searching for the best papers on: (1) loss functions for class imbalance and (2) linear/tubular structure segmentation.

**How to use:** Run `python scripts/elapsed_time.py` at start and end to measure your own 30 min. Aim for ~15 min on “core formulas and key papers,” then ~15 min on “implementation links and actionable list.”

---

## 1. Core loss functions (formulas and sources)

### 1.1 Generalized Dice loss (GDL) — Sudre et al. 2017

**Source:** Sudre, C.H. et al., *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations*, DLMIA/ML-CDS@MICCAI 2017. arXiv:1707.03237.
**Cited by:** ResUNet-a (and most imbalance segmentation work).

**Formula:**
- **Weight per class** (inverse volume): \( w_\ell = 1 / \big(\sum_i r_i^\ell\big)^2 \), where \( r_i^\ell \) is the ground-truth value for class \(\ell\) at pixel \(i\).
- **GDL** = \( 1 - \frac{\sum_\ell w_\ell \cdot (2\sum_i p_i^\ell r_i^\ell)}{\sum_\ell w_\ell \cdot (\sum_i p_i^\ell + \sum_i r_i^\ell)} \).

**Why it helps:** Reduces the bias of standard Dice toward large regions; small (lobe) class contributes more to the loss. Direct drop-in alternative to our current Dice/IoU in combined or ACL-style loss.

**Implementation:** Many PyTorch snippets exist; Papers with Code and PMC (PMC7610921) have references.

---

### 1.2 Boundary loss — Kervadec et al. (MIDL 2019 / MedIA 2021)

**Source:** Kervadec et al., *Boundary loss for highly unbalanced segmentation*, MIDL 2019; MedIA 2021.
**Cited by:** Boundary-aware segmentation; Aryal’s glacier work uses a related idea (masked Dice + boundary term).

**Idea:** Loss on **contour distance** (shape space) instead of region integrals. Expressed as a regional integral over softmax outputs, so it’s differentiable and fits standard N-D segmentation nets.

**Reported gains (with GDL):** up to ~8% Dice, ~10% Hausdorff, more stable training.

**Code:** **LIVIAETS/boundary-loss** (GitHub, MIT, PyTorch).
https://github.com/LIVIAETS/boundary-loss

**For us:** Add a boundary term alongside Dice or IoU (e.g. boundary + weighted Smooth L1) to stress boundary/localisation; fits our “blob in wrong place” issue.

---

### 1.3 Unified Focal loss (UFL) — Yeung et al. 2022

**Source:** Yeung et al., *Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation*, Computerized Medical Imaging and Graphics 2022. arXiv:2102.04525.

**Parameters:**
- **λ** (0–1): balance between distribution-based (CE-like) and region-based (Dice-like) terms.
- **γ**: focal correction (hard examples); lower γ = stronger focal effect.
- **δ** (0–1): balance false negatives vs false positives.

**Variants:** Symmetric (λ·L_mF + (1−λ)·L_mFT) and **asymmetric** (often best). Outperforms CE, Focal, Dice, Tversky, Focal Tversky, Combo on several medical datasets.

**For us:** Try UFL as main or auxiliary loss (e.g. binarize proximity at threshold); use asymmetric variant and tune γ low if our Focal tended to “predict constant.”

**Code:** e.g. tayden/unified-focal-loss-pytorch (PyTorch).

---

### 1.4 clDice (centerline Dice) — Shit et al. CVPR 2021

**Source:** Shit et al., *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021. arXiv:2003.07311.

**Idea:** Similarity on **skeleton/centerline** of the segmentation (and prediction). Preserves **connectivity** of tubular/linear structures (vessels, roads, neurons). Theoretically guarantees topology preservation up to homotopy equivalence. **soft-clDice** is differentiable (iterative min/max pooling + thresholding).

**Datasets:** 2D/3D vessels, roads, neurons. Better connectivity and graph similarity than standard Dice.

**Code:** **jocpae/clDice** (GitHub).
https://github.com/jocpae/clDice

**For us:** Lobes are elongated/linear; clDice could encourage connected centreline and reduce broken or scattered predictions. Try as auxiliary term (e.g. clDice on thresholded proximity) or as main loss for a binary-lobe experiment.

---

### 1.5 GapLoss and NeighborLoss (road segmentation, RS)

**GapLoss:** Designed for **road** segmentation in remote sensing; targets boundary/gap errors.
**Paper:** MDPI Remote Sensing 14(10):2422.
**Code:** Dabao55/GapLoss (PyTorch).
https://github.com/Dabao55/GapLoss

**NeighborLoss:** Uses **spatial correlation** (neighboring pixels) for RS segmentation.
**Paper:** IEEE (e.g. document 9437182).
**Code:** chinaericy/neighborloss (TensorFlow/Keras).

**For us:** Roads ≈ linear, thin, imbalanced; same family as lobes. GapLoss for boundary emphasis; NeighborLoss for spatial consistency (less speckle).

---

## 2. Distance transform and multi-task (ResUNet-a style)

**ResUNet-a** (Diakogiannis et al., ISPRS 2020): Multi-task order — **boundary → distance transform → segmentation mask → reconstruction**. Distance transform is an **explicit intermediate target**, like our proximity maps.

**Other refs:**
- *Distance transform regression for spatially-aware deep semantic segmentation* (arXiv:1909.01671): joint classification + distance regression.
- *How Distance Transform Maps Boost Segmentation CNNs* (PMLR 2020): distance maps as auxiliary supervision improve boundaries and imbalance.

**For us:** We already predict proximity (distance-like). Option: auxiliary **binary** head with GDL or boundary loss, or multi-task with distance head + segmentation head sharing encoder.

---

## 3. Tversky and Focal Tversky

**Tversky loss:** Generalises Dice with parameters for FP/FN trade-off; good for small foreground.
**Focal Tversky:** Adds focal weighting on hard examples.
**Refs:** Hashemi et al. (Tversky loss); Abraham / Focal Tversky U-Net (GitHub: nabsabraham/focal-tversky-unet).

UFL generalises these; if we try UFL we cover this direction. Optional: try Focal Tversky alone as another baseline.

---

## 4. Self-learning boundary-aware loss (Aryal glacier)

**Aryal et al.** (Boundary Aware U-Net, arXiv:2301.11454):
\( L_{\mathrm{SLBA}} = \frac{1}{2\alpha_1^2} L_{\mathrm{MDice}} + \frac{1}{2\alpha_2^2} L_{\mathrm{Boundary}} + |\ln(\alpha_1\alpha_2)| \).
\(\alpha_1,\alpha_2\) are **learned** (no manual tuning). Masked Dice + boundary term; debris-covered ice IoU 0% with CE → 35.9% with L_SLBA.

**Code:** Aryal007/glacier_mapping (GitHub).

**For us:** Learnable weights for Dice + boundary could replace fixed λ in ACL; glacier domain is close to lobes.

---

## 5. Surveys and comparative studies

- **Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook** (OpenReview): 25 loss functions, taxonomy, evaluation.
- **Comparative analysis of loss functions for foreground–background imbalance** (Springer 2022): Focal, Dice, Tversky, Mixed Focal compared.
- **Jadon (2020)** CIBCB: survey + **log-cosh Dice**; GitHub: shruti-jadon/Semantic-Segmentation-Loss-Functions (Keras).

---

## 6. Actionable list for our project (priority)

| Priority | Action | Source |
|----------|--------|--------|
| 1 | Implement **Generalized Dice loss** (Sudre formula with \(w_\ell = 1/(\sum r_i^\ell)^2\)) and compare to current Dice/IoU in combined loss. | §1.1 |
| 2 | Try **ACL-style** (Dice + Focal with λ) as in backlog; optionally use **GDL** instead of standard Dice in ACL. | Gully-ERFNet, §1.1 |
| 3 | Add **Boundary loss** (Kervadec) alongside GDL or Dice; use LIVIAETS/boundary-loss. | §1.2 |
| 4 | Try **UFL** (asymmetric) as main or auxiliary loss; tune γ. | §1.3 |
| 5 | Experiment with **clDice** (soft-clDice) for thresholded lobe mask to favour connectivity. | §1.4 |
| 6 | Test **GapLoss** or **NeighborLoss** for boundary/spatial consistency (road-style linear structures). | §1.5 |
| 7 | Consider **learnable** Dice+boundary weights (Aryal-style) instead of fixed λ. | §4 |
| 8 | Optional: **Distance transform** as auxiliary head (ResUNet-a style); we already have proximity. | §2 |

---

## 7. Implementation links (quick reference)

| Item | URL |
|------|-----|
| Generalized Dice (Sudre) | arXiv:1707.03237; PMC7610921 |
| Boundary loss | https://github.com/LIVIAETS/boundary-loss |
| clDice | https://github.com/jocpae/clDice |
| GapLoss | https://github.com/Dabao55/GapLoss |
| NeighborLoss | https://github.com/chinaericy/neighborloss |
| UFL PyTorch | e.g. tayden/unified-focal-loss-pytorch |
| Jadon loss survey (Keras) | https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions |
| Aryal glacier L_SLBA | https://github.com/Aryal007/glacier_mapping |

---

## 8. Time check

Run at start:
`python scripts/elapsed_time.py`

Run at end:
`python scripts/elapsed_time.py`

Elapsed = end time − start time. Aim for ~30 min of focused reading and note-taking; adjust with the script as you go.

---

## Session log (30-min bibliography crawl)

**T0 (session start):** 22:49:28 local.

### Chunk 1–2: Following clDice / topology-preserving and Boundary loss

**New papers ranked (from clDice / topology / boundary bibliographies):**

1. **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024) — MIC-DKFZ. For thin tubular structures (vessels, nerves, roads, cracks). **Multi-class** capable; **>90% less compute** than other topology losses** (CPU-based). GitHub: MIC-DKFZ/Skeleton-Recall (Apache-2.0). Strong candidate for lobe/linear: connectivity + efficient.
2. **Topograph** (Lux, Berger et al., ICLR 2025) — Graph-based; component graph encodes topology; **strict topological metric** (homotopy); **~5× faster** than persistent-homology methods. arXiv:2411.03228.
3. **Persistent homology / Betti loss** — Differentiable topological loss (Betti numbers); nick-byrne/topological-losses (2D/3D). Theoretically strong but heavier compute; Topograph and Skeleton Recall are faster alternatives.
4. **CoLeTra** (2025) — Data augmentation “disconnect to connect”: train on disconnected-appearance images with connected labels to improve topology without perfect labels.

**Boundary loss (Kervadec):** Confirmed combined with GDL gives ~8% Dice, ~10% Hausdorff on ISLES/WMH; contour-distance integral; LIVIAETS/boundary-loss.

**Next (planned):** Follow Skeleton Recall and road/gully-specific papers; rank their refs.

---

## 9. Sustained 30-minute session (bibliography chaining)

**Instructions:** Run `python scripts/elapsed_time.py` at **start** (T0) and again every ~5–10 min and at **end** (T_end). Keep doing rounds below until elapsed time ≥ 30 min. If only 1–2 min have passed, start the next round (another paper + bibliography rank).

**T0 (session start):** 2026-02-03 22:50:58 local
**T1 (check):** 2026-02-03 22:51:46 local (~1 min) → continue with more rounds.

### Round 1–2: Xu 2023 (road loss), Zhu 2024 (LinkNet), follow-up surveys

- **Xu et al. 2023** (IJ Appl Earth Obs Geoinf 116:103159): Comparative study of loss functions for road segmentation; GapLoss, cost-sensitive, **NeighborLoss** (spatial correlation). Full bibliography not fetched; related: **GapLoss** MDPI Remote Sensing 14(10):2422; **Azad et al. 2023** “Loss Functions in the Era of Semantic Segmentation” (arXiv:2312.05391) — 25 losses, taxonomy, GitHub YilmazKadir/Segmentation_Losses.
- **Zhu et al. 2024** “Asymmetric Non-Local LinkNet” (Int Soil Water Conserv Res 12(2):365–378): Gully mapping Northeast China; non-local attention for linear structures. Exact ref confirmed from our Gully-ERFNet bib; no new refs mined in this round.

### Round 3–4: Road topology, boundary/skeleton, GapLoss

- **Road-Topology Loss** (MDPI Remote Sensing 13:2080): Ordinal regression + road-topology loss for SAR road extraction; penalizes gaps and spurious segments; +11.98% IoU detection, +10.9% Quality vs LinkNet34/DLinkNet/DeepLabV3+. **Rank for us:** High — same linear/thin/imbalance as lobes.
- **GapLoss:** Yuan & Xu (2022); MDPI 14(10):2422. Skeleton endpoints + buffer weighting → weighted CE. PyTorch: **Dabao55/GapLoss** (MIT). **NeighborLoss:** IEEE 9437182; spatial correlation, weights boundary/small-object pixels; GitHub **chinaericy/neighborloss** (TensorFlow/Keras).
- **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024, arXiv:2404.03010): Thin tubular (vessels, roads, cracks). CPU tubed skeletonization + GPU soft recall; **multi-class**; >90% less compute. GitHub **MIC-DKFZ/Skeleton-Recall**.
- **Skea-Topo** (arXiv:2404.18539, IJCAI 2024): Skeleton-aware weighted loss + Boundary rectified term (BoRT); +7 pt VI on boundary segmentation. GitHub **clovermini/skea_topo**.

### Round 5–6: Azad 2023 survey, new gully papers, Lu 2024

- **Azad et al. 2023** (Reza Azad, Moein Heidary, Kadir Yilmaz et al.): 25 loss functions, novel taxonomy, evaluations on medical + natural images; **GitHub YilmazKadir/Segmentation_Losses** (PyTorch). **Bibliography rank for our project:** Use as “try-list” — Focal, Dice, boundary, Tversky, etc. Focal Boundary Dice (breast MRI) as boundary variant.
- **Multi-Scale Content-Structure** (MDPI Remote Sensing 16(19):3562, 2024): Gully extraction network; multi-scale content-structure features. Same domain as our lobe/gully literature.
- **Lu et al. 2024** (Remote Sensing 16(5):921): InSAR-refined DEM + relative elevation algorithm; Huangfuchuan Loess; F1 81.94%, +9.77% vs ASTER; ~28k gullies. Confirms DEM quality/refinement helps when resolution is sufficient.

### Round 7–8: NeighborLoss, thermokarst, Phinzi

- **NeighborLoss** (IEEE 9437182): Weights pixels by neighborhood affinity (boundary + small objects); TensorFlow/Keras **chinaericy/neighborloss**. Directly relevant for linear/spatial consistency.
- **Huang et al.** (Remote Sensing 10:2067 — 2018): Thermokarst landforms, DeepLab, 0.15 m UAV, F1 0.74. Linear erosion landforms; DL from optical.
- **Phinzi, Holb, Szabó** (e.g. MDPI Agronomy 11(2):333, 2021): Mapping permanent gullies, satellite + ML (RF, SVM); k-fold and bootstrapping; binary vs multiclass.

### Round 9–10: TopoAL, Bi-HRNet, DTU-Net, clDice, GDL

- **TopoAL** (Vasu et al., arXiv:2007.09084): Adversarial learning; discriminator label pyramid (multi-scale) for road topology; STE binarization. GitHub **kstepanov7/TopoAL**. RoadTracer SOTA.
- **Bi-HRNet** (MDPI Remote Sensing 14:1732): Node heatmap + bidirectional connectivity; multi-task (points, edges, direction). Topology-aware road extraction.
- **DTU-Net** (arXiv:2205.11115, IPMI 2023): Dual U-Net (texture net + topology net); **triplet loss** for topological similarity (false splits / missed splits); curvilinear (vessels, neurons); no prior topology.
- **clDice** (Shit et al., CVPR 2021): soft-clDice differentiable; Tprec/Tsens on skeletons; jocpae/clDice, dmitrysarov/clDice.
- **Generalized Dice** (Sudre 2017, arXiv:1707.03237): Class weight \(w_\ell \propto 1/(\sum r_i^\ell)^2\); gravitino/generalized_dice_loss, pytorch-3dunet.

### Round 11–14 (continued)

- **Focal Boundary Dice** (J Cancer 2023, PMC10088889): Boundary + small-object emphasis for breast tumor MRI.
- **Zero-shot ephemeral gully 2025** (arXiv:2503.01169): VLMs; >70% acc, ~80% F1; first public ephemeral-gully dataset; Nov 2025 weakly supervised 18k+ images, noise-aware loss.
- **TopoRF-Net** (MDPI Sensors 25(24):7428): Multi-receptive field + connectivity-aware decoding + topology-aware loss; DeepGlobe IoU 69.76%, F1 82.18%.
- **BT-RoadNet** (ISPRS 2020): Coarse-to-fine, boundary + topology; GitHub fightingMinty/BT-RoadNet.
- **nnU-Net**: DiceCE; small objects → instance-wise (CC-DiceCE, blob), PM Dice; GDL for rebalancing.
- **Azad 2023 full 25 losses:** Pixel (4): CE, TopK, Focal, DMCE. Region (10): Dice, Log-Cosh Dice, Wasserstein Dice, IoU, Lovász-Softmax, Tversky, Focal Tversky, Sensitivity-Specificity, RMI, Robust T-Loss. Boundary (8): Boundary (Kervadec), Hausdorff, Boundary-aware, Active Boundary, InverseForm, Conditional Boundary, Boundary DoU, Region-wise. Combo (3): Combo, Exp-Log, UFL. **Top 5 for us:** Tversky/Focal Tversky, Jaccard, Boundary+GDL, UFL, Combo/Log-Cosh Dice. Also TAFL (topology), pixel-wise triplet (boundary).

### Bibliography ranks (papers to mine next when continuing)

| Paper | Why rank | Next action |
|-------|----------|-------------|
| Azad et al. 2023 | 25 losses in one place | ✅ Done Round 14 — see full list and top 5 below. |
| Road-Topology (SAR ordinal) | Connectivity + linear | Get full ref list; compare to GapLoss, NeighborLoss. |
| Skeleton Recall (ECCV 2024) | Multi-class, efficient | Mine its refs for other tubular losses. |
| TopoAL | Multi-scale topology | Check citing papers for road/lobe. |
| Lu 2024 InSAR DEM | DEM refinement | Compare to our DEM; slope-of-slope/curvature. |

### Rabbit hole (next 30 min): from one of the above

1. ~~Open **Azad 2023** arXiv HTML (2312.05391)~~ ✅ Done Round 14. (Rank top 5: “small foreground, linear, imbalance.”
2. Open **Skeleton Recall** (MIC-DKFZ) paper; list key refs (clDice, boundary, etc.); pick one and rank *its* bibliography.
3. **Zero-shot ephemeral gully 2025** (arXiv:2503.01169): Dataset/loss details; add to data options if relevant. 4. **Road-topology loss** (MDPI 13:2080): Full ref list; compare to GapLoss, NeighborLoss for try-order.

---

*Document created from bibliography following and web search; key formulas and code links verified. Sustained session §9 added 2026-02-03; Rounds 11–14 added. Run `python scripts/elapsed_time.py` at end to confirm 30 min total.*
