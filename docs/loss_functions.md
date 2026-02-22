# Loss functions

Training uses **proximity maps** (values 0–20): 0 = background, 1–19 = distance to lobe, 20 = lobe center. The model predicts a continuous value per pixel. The loss you choose in `configs/training_config.yaml` (`training.loss_function`) affects both **where** the model puts high values (localization) and **how close** predictions are to the target (regression).

All losses are implemented in `src/models/losses.py`.

---

## 1. `smooth_l1` — Basic Smooth L1 (Huber)

**What it does:** Standard regression loss. Smooth L1 (Huber): quadratic for small errors, linear for large errors. Same weight for every pixel.

**Formula:** For each pixel, if `|pred - target| < beta`: loss = `0.5 * (pred - target)² / beta`; else: loss = `|pred - target| - 0.5 * beta`. Then mean over pixels.

**Config:** `training` → (optional) no extra params; Smooth L1 uses `beta=1.0` by default in code.

**When to use:** Balanced data, or when you want a simple baseline. With strong imbalance (e.g. 86% background), the model often predicts low everywhere and ignores lobes.

---

## 2. `weighted_smooth_l1` — Weighted Smooth L1

**What it does:** Same as Smooth L1, but **lobe pixels** (target ≥ `lobe_threshold`) get a higher weight so the model pays more attention to them.

**Formula:** Per-pixel Smooth L1 as above, then multiply by a weight: weight = `lobe_weight` where target ≥ `lobe_threshold`, else 1. Mean over pixels.

**Config:** `training.lobe_weight` (e.g. 5.0), `training.iou_threshold` or lobe threshold (e.g. 5.0) for which pixels count as lobe.

**When to use:** You want regression (match values) but with class imbalance; raising `lobe_weight` pushes the model to fit lobe pixels better. Does not explicitly optimize overlap (IoU).

---

## 3. `dice` — Dice loss

**What it does:** Treats the task as **segmentation**. Predictions and targets are binarized with a threshold (e.g. pred ≥ 5 → 1, else 0). Loss = 1 − Dice coefficient. Optimizes overlap between predicted and target “lobe” regions; does not care about exact proximity values.

**Formula:** `pred_binary = (pred >= threshold)`, `target_binary = (target >= threshold)`. Dice = `(2 * intersection + smooth) / (pred_sum + target_sum + smooth)`. Loss = 1 − Dice.

**Config:** `training.iou_threshold` (used as the binarization threshold, e.g. 5.0).

**When to use:** When you care mainly about **where** the lobe is (binary mask), not the exact proximity values. Gradient is zero at the threshold (hard binarization), so training can be less smooth than soft_iou.

---

## 4. `iou` — IoU loss (Jaccard)

**What it does:** Same idea as Dice but with **IoU (Jaccard)**. Binarizes pred and target with a threshold. Loss = 1 − IoU. Directly optimizes intersection-over-union of the “lobe” set.

**Formula:** `pred_binary = (pred >= threshold)`, `target_binary = (target >= threshold)`. IoU = `(intersection + smooth) / (union + smooth)`. Loss = 1 − IoU.

**Config:** `training.iou_threshold` (binarization threshold).

**When to use:** When you want to maximize overlap (IoU) of the lobe region. Same caveat as Dice: hard threshold gives zero gradient at the boundary; consider `soft_iou` for smoother training.

---

## 5. `soft_iou` — Soft IoU (smooth gradients)

**What it does:** IoU loss with **soft** binarization via sigmoid instead of a hard threshold. Predictions and targets are turned into soft masks: `sigmoid((value - threshold) * temperature)`. Gradients are non-zero everywhere, so the model can learn the boundary more smoothly.

**Formula:** `pred_soft = sigmoid((pred - threshold) * temperature)`, same for target. Soft IoU = intersection / union with these soft masks. Loss = 1 − soft IoU.

**Config:** `training.iou_threshold`, and in code: `temperature` (default 1.0; higher = sharper transition).

**When to use:** When you want IoU-like optimization but with smoother gradients. Good default when you care about localization and want stable training.

---

## 6. `encouragement` — Encouragement loss

**What it does:** Only looks at **lobe pixels** (target ≥ `lobe_threshold`). Two terms: (1) MSE on lobe pixels (match target values), (2) penalty when prediction is **below** the threshold (encourages the model to predict at least the threshold on lobe pixels).

**Formula:** On lobe pixels: `loss = MSE(pred, target) + encouragement_weight * mean((threshold - pred)²)` for pred &lt; threshold.

**Config:** `training.lobe_threshold` (or iou_threshold), `training.encouragement_weight` (e.g. 2.0 or 10.0).

**When to use:** When the model tends to predict 0 everywhere and you want to “push” it to predict high on lobe pixels. Does not optimize background or global MAE; use with care on very imbalanced data.

---

## 7. `focal` — Focal loss (regression)

**What it does:** Regression loss that (1) **balances** background vs lobe via `alpha`, and (2) **down-weights easy examples** (small error) and focuses on hard ones (large error) via `gamma`. Reduces the tendency to minimize loss by predicting background everywhere.

**Formula:** Per pixel: `error_normalized = |pred - target| / max_error`, `modulating = (error_normalized)^gamma`, `alpha_mask` = alpha on lobe pixels and (1−alpha) on background. Loss = mean(alpha_mask * modulating * (pred - target)²).

**Config:** `training.focal_alpha` (e.g. 0.25–0.75), `training.focal_gamma` (e.g. 2.0; avoid 4+ or gradients can flatten), `training.iou_threshold` or lobe_threshold for “lobe” in alpha.

**When to use:** Strong class imbalance (e.g. 86% background). Prefer gamma ≈ 2; high gamma can lead to a flat loss and mean prediction stuck around 4–6. See `docs/focal_loss_explained.md` for details.

---

## 8. `combined` — IoU + Weighted Smooth L1

**What it does:** Sum of two terms: (1) **IoU loss** (or soft IoU) for localization, (2) **Weighted Smooth L1** for regression. Trades off overlap (where the lobe is) and value accuracy (how close pred is to target).

**Formula:** `loss = iou_weight * iou_loss(pred, target) + regression_weight * weighted_smooth_l1(pred, target)`.

**Config:** `training.iou_weight`, `training.regression_weight`, `training.use_soft_iou` (soft vs hard IoU), `training.iou_threshold`, `training.lobe_weight`, `training.lobe_threshold` (for the regression part).

**When to use:** When you want both good **localization** (IoU) and good **value fit** (regression). Often a good default; tune the two weights to emphasize one or the other.

---

## 9. `acl` — Adaptive Correction Loss (Dice + Focal)

**What it does:** Combines **Dice loss** (overlap) and **Focal loss** (hard-example focus) with a tunable weight: `loss = λ·Dice + (1−λ)·Focal`. From Gully-ERFNet (Li et al., IJDE 2025) for linear structures, severe imbalance, and label noise.

**Formula:** `loss = acl_lambda * dice_loss(pred, target) + (1 - acl_lambda) * focal_loss(pred, target)`.

**Config:** `training.acl_lambda` (0–1; default 0.5 = equal Dice and Focal), `training.iou_threshold` (binarization for Dice; same threshold used for Focal lobe weighting), `training.focal_alpha`, `training.focal_gamma` (passed to the Focal component).

**When to use:** Linear/elongated structures and strong foreground–background imbalance (e.g. gullies, lobes). Tune `acl_lambda`: higher = more overlap (Dice), lower = more regression/hard-example focus (Focal).

---

## Summary

| Loss                  | Optimizes              | Best for                          |
|-----------------------|------------------------|-----------------------------------|
| smooth_l1             | Value match (all same) | Simple baseline, balanced data    |
| weighted_smooth_l1    | Value match (lobe up)  | Regression + imbalance            |
| dice                  | Overlap (binary)       | Segmentation, overlap              |
| iou                   | Overlap (binary)       | IoU; hard threshold                |
| soft_iou              | Overlap (soft)         | IoU with smooth gradients         |
| encouragement         | Lobe pred ≥ threshold | Pushing model to predict on lobes |
| focal                 | Value + focus on hard  | Extreme imbalance                 |
| combined              | IoU + regression       | Localization and values together  |
| acl                   | Dice + Focal (λ)       | Linear structures, imbalance (Gully-ERFNet) |

For more on Focal Loss (formula, gamma, why mean prediction can get stuck), see [focal_loss_explained.md](focal_loss_explained.md).
