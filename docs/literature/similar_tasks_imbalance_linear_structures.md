# Literature: Similar Tasks (Linear/Elongated Structures, Severe Imbalance)

Short list of papers that may be useful for lobe detection: elongated/linear structures and strong background–target imbalance. Add more as you find them.

---

## Dataset size comparison

To compare fairly, we use **number of training images/tiles** (samples per epoch) and **image/tile size** (pixels). Some papers report "number of labels" or "number of scenes"; where possible we use **training set size** and **spatial size** so we can compare total pixels per epoch.

| Source | Train (samples) | Val (samples) | Test | Tile/patch size | Total train pixels (approx) |
|--------|------------------|---------------|------|------------------|-----------------------------|
| **Our project (production)** | **1,854** tiles | **397** tiles | ~15% of data | 256×256 | 1,854 × 256² ≈ **121 M** |
| **ResUNet-a (Potsdam)** | varies by cropping | — | — | 38 scenes of **6000×6000**; often cropped to patches | 38 × 6000² = **1.37e9** raw; effective train N depends on patch sampling |
| **Boundary-Aware U-Net (Aryal)** | **333** sub-images | **68** | **98** | **512×512** | 333 × 512² ≈ **87 M** |
| **Unified Focal Loss (UFL) – DRIVE** | **20** images | — | 20 | 768×584 | 20 × (768×584) ≈ **9 M** |
| **Unified Focal Loss – CVC-ClinicDB** | **612** images | — | — | 288×384 | 612 × (288×384) ≈ **68 M** |
| **Unified Focal Loss – BraTS20** | **369** volumes (3D) | — | — | 3D MRI | Different modality |
| **Gully-ERFNet (Li et al. 2025)** | **~5,120** train pairs | **~640** | **~640** | **512×512** | ~5,120 × 512² ≈ **1.34e9** |

**Takeaways:**

- **Our project** has **more training samples** (1,854) than Boundary-Aware U-Net (333) and far more than UFL’s DRIVE (20). We have **comparable or larger** total train pixels than Aryal (121 M vs 87 M) and CVC-ClinicDB (121 M vs 68 M).
- **ResUNet-a** uses very large raw images (38 × 6000×6000); effective training size depends on how many overlapping/non-overlapping patches they extract per epoch. In many setups, that still yields thousands to tens of thousands of patches, so we are in a similar order of magnitude.
- **UFL** is validated on very small 2D datasets (20–612 images); their loss is designed to help when data are limited and imbalance is extreme. So **dataset size alone is unlikely to explain our lack of improvement** — we have more training samples than several of their settings. The difference may be **task difficulty** (proximity regression vs binary segmentation), **label quality**, or **loss/objective** (e.g. adding UFL or a mean-matching term).

*Units: "samples" = number of images/tiles/patches used per epoch; "total train pixels" = train samples × height × width (ignoring overlap).*

---

## Our project: loss function and results

**Task:** Proximity regression — predict a continuous value per pixel (0–20): 0 = background, 1–19 = distance to lobe, 20 = lobe center. Evaluated with MAE (primary), IoU (thresholded), and improvement over a “predict 0” baseline.

**Why proximity maps instead of binary boundaries:** We could also use a binary target (lobe vs non-lobe). We started from **1-px-wide lobe boundary lines**, but that was not very precise and subject to differences between interpreters. We switched to **proximity maps** to make the task easier for the model: they encode not only the boundary but also "more or less where other people might draw those lines," reducing sensitivity to exact line placement and interpreter variability. Binary (e.g. thresholded proximity or redrawn masks) remains an option for experiments (e.g. ACL / Dice+Focal from Gully-ERFNet) or as an auxiliary head.

**Loss functions available** (see `docs/loss_functions.md`): `smooth_l1`, `weighted_smooth_l1`, `dice`, `iou`, `soft_iou`, `encouragement`, `focal`, `combined`. Recent production and tuning runs use mainly **combined** (IoU + weighted Smooth L1) or **focal**; combined was selected by Optuna as best for val loss.

**Combined loss:** Weighted sum of (1) **IoU loss** (1 − IoU on binarized pred/target with a threshold) and (2) **weighted Smooth L1** (Smooth L1 with higher weight on lobe pixels). Balances overlap and regression; hyperparameters (e.g. IoU weight, lobe weight, threshold) come from `configs/training_config.yaml` or from Optuna tuning.

**Results so far (representative run, 600 epochs, combined loss, HP from tuning):**

| Metric | Value |
|--------|--------|
| Best val loss | **3.42** (at ~epoch 26) |
| Best val MAE | **2.04** |
| Best val IoU | **~0.13** |
| Baseline MAE (“predict 0”) | **1.39** |
| Improvement over baseline | **Negative** (val MAE &gt; baseline) |

Training loss decreases (e.g. 3.37 → 0.86) while validation loss and val MAE plateau after ~26–30 epochs; the model has not yet achieved validation MAE better than the “predict 0” baseline. See `docs/daily_diary/2026-02-02.md` and `docs/no_learning_improvement_causes.md` for interpretation and next steps.

---

## 1. ResUNet-a: Deep Learning for Semantic Segmentation of Remotely Sensed Data

**Authors:** Diakogiannis, Waldner, Caccetta, Wu
**Venue:** ISPRS Journal of Photogrammetry and Remote Sensing (2020)
**Links:** [arXiv:1904.00592](https://arxiv.org/abs/1904.00592) · [Papers with Code](https://paperswithcode.com/paper/resunet-a-a-deep-learning-framework-for) · [GitHub: Aidence/resuneta](https://github.com/Aidence/resuneta)

**Dataset:** ISPRS 2D Potsdam — **38 tiles** of **6000×6000** pixels each (aerial, urban). Train/val/test split varies by paper; effective training is usually many overlapping or non-overlapping patches from these 38 scenes (order of magnitude: thousands to tens of thousands of patches per epoch).

**Loss function:** **Generalized Dice loss** — novel variant analysed in the paper; designed for highly imbalanced classes with good convergence. Multi-task training: boundaries → distance transform → segmentation mask → reconstruction (each conditioned on the previous).

**Results:** **Average F1 92.9%** over all classes on ISPRS 2D Potsdam; state-of-the-art at publication. Per-class F1 and IoU reported; performance strong on small/thin structures (roads, etc.) thanks to the loss and multi-task setup.

**Why relevant:**
- Remote sensing semantic segmentation with **high class imbalance** (small foreground vs large background).
- Uses a **Generalized Dice loss** variant designed for imbalanced classes and reports good convergence.
- **Multi-task setup**: infers boundaries, **distance transforms of segmentation masks**, then segmentation mask, then reconstruction — so **distance/proximity is an explicit intermediate target**, similar to our proximity maps.
- Architecture: U-Net + residual connections + atrous convolutions + PSP; applicable to aerial/satellite imagery.

**What we could adopt:**
- **Generalized Dice loss** formulation and comparison to standard Dice/CE (they analyse several variants).
- **Auxiliary distance-transform target**: train with an auxiliary head or loss on distance-to-boundary (like our proximity) in addition to binary mask; could stabilise learning.
- Multi-task conditioning (boundary → distance → mask) as a possible two-stage or auxiliary-task design.

---

## 2. Boundary Aware U-Net for Glacier Segmentation

**Authors:** Aryal et al.
**Venue:** Northern Lights Deep Learning Workshop (2023)
**Links:** [arXiv:2301.11454](https://arxiv.org/abs/2301.11454) · [GitHub: Aryal007/glacier_mapping](https://github.com/Aryal007/glacier_mapping)

**Dataset:** Hindu Kush Himalaya (Landsat 7). **141 cells** → cropped into **333 train** sub-images, **68 val**, **98 test**. Each sub-image **512×512**; sub-images with &lt;10% glacier pixels discarded to reduce imbalance. Labels: clean ice ~22%, debris-covered ~2.4%, background ~72%, masked ~3%. So **fewer training samples** (333) than us (1,854), similar patch size (512 vs our 256).

**Loss function:** **Self-learning boundary-aware loss (L_SLBA)** — weighted combination of **masked Dice loss** and **boundary loss**; weights α₁, α₂ learned during training (no manual tuning). Compared to: cross-entropy (L_CE), fixed-α combined loss (L_Combined). Formula: L_SLBA = (1/2α₁²)·L_MDice + (1/2α₂²)·L_Boundary + |ln(α₁·α₂)|.

**Results:** **Clean glacial ice:** Precision ~81.6%, Recall ~80.8%, **IoU ~68.2%**. **Debris-covered glacial ice:** best with L_SLBA — Precision ~52%, Recall ~53.8%, **IoU ~35.9%**; standard U-Net with L_CE gives **IoU 0%** on debris (model fails to detect). Boundary-aware loss clearly improves debris segmentation over Dice alone.

**Why relevant:**
- **Glacier segmentation** in remote sensing (Hindu Kush Himalaya): clean ice and **debris-covered glacial ice** vs background — direct domain analogue to solifluction lobes.
- **Severe imbalance**: debris-covered ice is hard to distinguish from moraines/terrain (similar to our lobe vs background).
- They introduce a **self-learning boundary-aware loss** that outperforms standard Dice loss for this task.
- Band importance (red, SWIR, NIR) for glacier mapping; we use RGB + DEM + slope, so band/feature analysis is relevant.

**What we could adopt:**
- **Boundary-aware loss** design: explicit term that emphasises boundary pixels or boundary consistency; could help with localisation and reduce “blob in wrong place”.
- Architecture tweaks (modified U-Net for large-scale segmentation) and training setup.
- If code is available, inspect how the boundary-aware loss is implemented and whether it can be combined with our proximity regression.

---

## 3. Unified Focal Loss: Generalising Dice and Cross-Entropy to Handle Class Imbalance (Medical) + TransUNet + UFL for Segmentation

**References:**
- **Unified Focal Loss:** Yeung et al., *Medical Image Analysis* (2022) — [arXiv:2102.04525](https://arxiv.org/abs/2102.04525); [GitHub: mlyg/unified-focal-loss](https://github.com/mlyg/unified-focal-loss), [tayden/unified-focal-loss-pytorch](https://github.com/tayden/unified-focal-loss-pytorch).
- **TransUNet + UFL for class-imbalanced segmentation:** Springer *Artificial Life and Robotics* (2023) — [Springer](https://link.springer.com/article/10.1007/s10015-023-00919-2) (application to segmentation with imbalance).

**Dataset (UFL paper):** Five medical datasets — **DRIVE** 20 train / 20 test images (768×584), **CVC-ClinicDB** 612 images (288×384), **BUS2017**, **BraTS20** 369 volumes (3D), **KiTS19** (3D). So **much smaller** 2D train sets (20–612 images) than ours (1,854 tiles); UFL is shown to help when data are limited and imbalance is extreme.

**Loss function:** **Unified Focal loss (UFL)** — combines modified Focal loss (suppresses background) and modified Focal Tversky loss (enhances rare class). **Symmetric** (λ·L_mF + (1−λ)·L_mFT) and **asymmetric** (L_maF + L_maFT) variants. Hyperparameters: λ=0.5, δ=0.6, γ tuned (e.g. 0.5 for 3D). Compared to: CE, Focal, Dice, Tversky, Focal Tversky, Combo. UFL consistently **outperforms** these on all five datasets.

**Results (UFL paper, U-Net, best reported):**

| Dataset        | Metric   | Best (UFL Asym) | Note                          |
|----------------|----------|------------------|-------------------------------|
| **CVC-ClinicDB** | DSC / IoU | **0.909** / **0.851** | Polyp; 9.3% foreground        |
| **DRIVE**       | DSC / IoU | **0.803** / **0.671** | Vessels; 8.7% foreground      |
| **BUS2017**     | DSC / IoU | **0.824** / **0.731** | Breast tumour; 4.8% foreground |
| **BraTS20**     | DSC / IoU | **0.787** / **0.683** | Enhancing tumour (3D); 0.2% foreground |
| **KiTS19**      | Kidney DSC / Tumour DSC | **0.943** / **0.634** | 3D multi-class; tumour ~0.2%  |

UFL (asymmetric) beats CE, Focal, Dice, Tversky, Focal Tversky, and Combo on all tasks; Focal loss in particular performs **worse** than CE on some datasets (e.g. CVC-ClinicDB, BUS2017), while UFL avoids that and improves recall–precision balance.

**Why relevant:**
- **Unified Focal Loss (UFL)** generalises Dice and cross-entropy into one framework and is tuned for **extreme class imbalance** (small foreground, large background).
- Validated on medical imaging (vessels, lesions, etc.) — often **elongated or sparse structures** with strong imbalance.
- Combined with TransUNet (or other backbones), it extracts **small regions of minor classes without increasing false positives** — directly addresses our “model worse than baseline” / “predict constant” behaviour.
- PyTorch implementations available; easier to try as an alternative to our current Focal or Combined loss.

**What we could adopt:**
- **Replace or complement** our Focal Loss with **Unified Focal Loss** (or a variant) and compare val MAE / improvement over baseline.
- Hyperparameter ranges and tuning tips from the paper/supplement.
- If we keep a regression head (proximity), we could use UFL for an auxiliary binary segmentation head and keep Smooth L1 for proximity, or explore a UFL-style formulation for regression (if any exists in follow-up work).

---

## 4. Gully-ERFNet: extracting erosion gullies (Li et al., IJDE 2025)

**Authors:** Qingyao Li, Jiuchun Yang, Jiaqi Wang, Zhi Li, Jianwei Fan, Liwei Ke, Xue Wang
**Venue:** International Journal of Digital Earth (2025)
**Links:** [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2494074) · DOI: 10.1080/17538947.2025.2494074
**Local copy:** `docs/external_documents_articles/Gully-ERFNet a novel lightweight deep learning model for extracting erosion gullies in the black soil region of Northeast China.pdf`

**Dataset:** GF-2 satellite imagery (0.8 m after fusion), Hailun Basin and Zake-Keyin Basin, Northeast China. **512×512** pixel blocks; augmented to **6,400** image–label pairs; split **8:1:1** (train ~5,120, val ~640, test ~640). Imagery from bare-soil period (November) to reduce vegetation occlusion. Labels from OBIA + manual refinement; cross-validated with national survey and aerial imagery. No high-resolution DEM required; 12.5 m DEM was tested as extra channel and **did not improve** (slight decrease).

**Loss function:** **Adaptive Correction Loss (ACL)** — weighted combination of **Dice loss** and **Focal loss**:
ACL = λ·Loss_Dice + (1−λ)·Loss_Focal.
λ is adjusted during training to balance the two terms. Dice emphasises boundary/overlap; Focal addresses **class imbalance** (gully vs non-gully) and **label noise**. The paper reports that ACL yields faster convergence and lower final loss than cross-entropy alone.

**Architecture:** **Gully-ERFNet** — ERFNet encoder–decoder + **Squeeze-and-Excitation (SE)** attention (channel recalibration) + **Pyramid Pooling Module (PPM)** between encoder and decoder for multi-scale context. Lightweight, aimed at small/linear targets.

**Results:** Gully-ERFNet: **Precision 87.54%**, **Recall 76.24%**, **F1 81.50%**, **IoU 68.80%**, Kappa 0.826. Outperforms RF (F1 63.50%, IoU 46.52%), U-Net (F1 70.67%, IoU 54.64%), DeeplabV3+ (F1 79.47%, IoU 65.95%), and baseline ERFNet (F1 76.84%, IoU 62.40%). Post-processing (remove_small_objects / remove_small_holes at 1000 px, road buffer exclusion) further improves metrics. DEM fusion slightly reduced precision (87.54% → 86.81%).

**Why relevant:**
- **Linear/elongated structures** (erosion gullies) in remote sensing, with **severe class imbalance** (gully pixels rare) and confusion with roads/ditches — close analogue to our lobe vs background.
- **ACL = Dice + Focal** is a simple, interpretable combo for imbalance and noise; we use **combined (IoU + weighted Smooth L1)** — trying a **Dice + Focal** variant (or ACL-style λ schedule) could be useful.
- **No high-res DEM needed** — they show optical-only can work; we use DEM/slope, but their finding supports that improving loss and architecture may matter more than extra channels if DEM is coarse.
- **Post-processing** (morphology + road masking) is explicit; we could consider similar cleanup for lobe maps.

**What we could adopt:**
- **ACL-style loss**: Dice (or soft IoU) + Focal with a learnable or scheduled λ; compare to our current combined loss on val MAE and improvement over baseline.
- **SE + PPM** in our encoder–decoder if we want a lightweight multi-scale option without changing backbone.
- Post-processing pipeline (small-object/hole removal, optional mask from roads/artifacts) for final lobe maps.

---

## Summary

| Paper | Domain | Main idea | What to try first |
|-------|--------|-----------|--------------------|
| **ResUNet-a** | Remote sensing (general) | Generalized Dice + **distance transform** as auxiliary target | Generalized Dice variant; auxiliary proximity/distance head or loss |
| **Boundary-Aware U-Net** | Glacier segmentation | **Boundary-aware loss** for debris-covered ice | Boundary-aware loss term for better localisation |
| **Unified Focal Loss** | Medical (vessels, etc.) / segmentation | **UFL** for extreme imbalance; less “predict constant” | Replace or add UFL; compare val MAE vs baseline |
| **Gully-ERFNet (Li et al. 2025)** | Gully extraction (remote sensing) | **ACL** (Dice + Focal); SE + PPM; no high-res DEM | ACL-style loss (Dice + Focal); optional SE/PPM; post-processing |

---

## Next steps

- Read the **Generalized Dice** section and loss equations in ResUNet-a; implement or adapt one variant and compare to our Combined loss.
- Get **Boundary-Aware U-Net** paper/code; extract the boundary-aware loss formula and implement a variant (e.g. boundary-weighted term) alongside our current loss.
- Add **Unified Focal Loss** (PyTorch) to the codebase; run the same config with UFL instead of Focal or Combined and monitor val MAE and improvement over baseline.
- Try **ACL-style loss** (Dice + Focal with λ) from Gully-ERFNet; compare to our combined (IoU + weighted Smooth L1) on val MAE and improvement over baseline.
- **Deeper dive:** Formulas, code links, and priority list (GDL, Boundary loss, UFL, clDice, GapLoss): `docs/literature/bibliography_deep_dive_30min.md`.

*Last updated: 2026-02-02. Add new papers and “Verified / Ruled out” notes as you go.*
