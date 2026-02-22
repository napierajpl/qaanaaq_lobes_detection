# Focal Loss (regression) — explained

Focal Loss in this project is a **regression** variant of the original Focal Loss (Lin et al., 2017, for object detection). It is used to train the proximity-map model when there is strong class imbalance (e.g. ~86% background vs ~14% lobe pixels).

---

## 1. The problem it addresses

With standard MSE or L1:

- **Background pixels** (target 0) are the vast majority.
- **Lobe pixels** (target 1–20) are a small minority.

The optimizer can reduce the loss a lot by just predicting “0 everywhere”: MSE drops because most pixels are correct. The model then pays little attention to lobe pixels and underfits them. Focal Loss is designed so that:

1. **Lobe pixels matter more** (class balance via `alpha`).
2. **Hard examples matter more** (focus via `gamma`): pixels where the prediction is still wrong get more weight; pixels that are already correct get less weight.

---

## 2. Original Focal Loss (classification)

In the **classification** setting (e.g. object detection):

- **Easy example**: model predicts high probability for the correct class → loss is small and is **down-weighted**.
- **Hard example**: model predicts low probability for the correct class → loss is larger and is **up-weighted** (relative to easy ones).

So the loss “focuses” on hard examples. The formula uses a factor \((1 - p_t)^\gamma\): when the model is confident and correct (\(p_t\) high), \((1 - p_t)^\gamma\) is small, so that example contributes little. When the model is wrong or uncertain (\(p_t\) low), the factor is larger.

---

## 3. Our Focal Loss for regression

We have **continuous** targets (proximity 0–20), so “easy” and “hard” are defined by **prediction error**:

- **Easy**: prediction close to target (small error).
- **Hard**: prediction far from target (large error).

We want to down-weight easy pixels and focus on hard ones, while still balancing background vs lobe via `alpha`.

### Formula (per pixel)

```
error             = |pred - target|
error_normalized  = clamp(error / max_error, 0, 1)   # max_error = max(target.max(), 20)
modulating_factor = (error_normalized)^gamma
alpha_mask        = alpha for lobe pixels (target >= lobe_threshold), else (1 - alpha)

focal_loss_per_pixel = alpha_mask * modulating_factor * (pred - target)²
```

Final loss = mean (or sum) over all pixels.

So:

- **Base term**: MSE, \((pred - target)^2\).
- **Modulating factor**: \((error\_normalized)^\gamma\). Small error → factor near 0 (easy, down-weighted). Large error → factor near 1 (hard, full weight).
- **Alpha mask**: lobe pixels get weight `alpha`, background get `(1 - alpha)` so you can emphasize lobes.

---

## 4. Parameters

| Parameter        | Role | Typical values |
|-----------------|------|-----------------|
| **alpha**       | Class balance. Lobe pixels (target ≥ `lobe_threshold`) get weight `alpha`, background get `(1 - alpha)`. | 0.25 (more weight on background), 0.5 (balanced), 0.75 (more weight on lobes) |
| **gamma**       | Focus on hard examples. Loss weight per pixel is proportional to `(error_normalized)^gamma`. Higher gamma → easy examples weighted less. | 0 ≈ MSE; 2 = standard focal; 4+ = very strong focus, can flatten gradients |
| **lobe_threshold** | Pixels with target ≥ this are “lobe” for the alpha mask. | 5.0 (or match your IoU threshold) |

- **alpha &lt; 0.5**: background weighted more than lobe.
- **alpha = 0.5**: balanced.
- **alpha &gt; 0.5**: lobe weighted more than background (common when lobe pixels are rare).
- **gamma = 0**: modulating factor = 1 everywhere → same as weighted MSE.
- **gamma = 2**: moderate down-weighting of easy examples.
- **gamma = 4**: strong down-weighting; medium errors (e.g. pred=6, target=0) get very small weight → gradients can become tiny and training can stall around a constant prediction.

---

## 5. Intuition with numbers

Assume target range 0–20, so `max_error = 20`.

**Background pixel, target = 0**

- Pred = 0 → error = 0 → modulating = 0 → no gradient (already correct).
- Pred = 6 → error = 6 → error_norm = 0.3 → modulating = \(0.3^\gamma\).
  - gamma=2: \(0.3^2 = 0.09\) → gradient scaled by 0.09.
  - gamma=4: \(0.3^4 \approx 0.008\) → gradient scaled by 0.008 (very small).

So **high gamma** makes the gradient that would pull “6” down toward 0 very small. The loss is flat around “predict 6” on background.

**Lobe pixel, target = 10**

- Pred = 10 → error = 0 → modulating = 0 (easy).
- Pred = 6 → error = 4 → error_norm = 0.2 → modulating = \(0.2^\gamma\); still small for high gamma.

So with high gamma, **both** “predict 6 on background” and “predict 6 on lobe” can sit in a flat region: gradients are small, and the model may not move mean predictions toward the true mean (e.g. 1.4).

---

## 6. Why the model can get stuck predicting ~6

- The **optimal constant** under Focal Loss (with alpha) is a compromise between “match 0 on background” and “don’t underestimate lobes.” That constant is often **above** the global mean (e.g. in the 4–6 range).
- With **high gamma** (e.g. 4), gradients for “medium” errors (like 6 on background) are tiny, so the loss is flat around “predict 6” and the optimizer doesn’t get a strong signal to move the mean prediction down toward the true mean.

So the model is not “refusing” to predict the mean; the loss and gamma are such that (1) the best constant is above the mean and (2) gradients are too small to push the mean down. See `docs/run_3c5caa5b_analysis.md` for more detail.

---

## 7. Practical tuning

- **Start with gamma = 2** (or 1). Avoid gamma ≥ 4 unless you have a good reason; it often flattens gradients and encourages a constant prediction.
- **alpha**: 0.5 for balance; 0.75 if lobes are very rare and you want to force the model to care about them (watch for mean prediction drifting up).
- **lobe_threshold**: usually match your “lobe” definition (e.g. 5 if proximity ≥ 5 means “in lobe”).
- If the model still predicts a constant above the true mean, consider: lowering gamma, adding a small MAE or mean-matching term, or trying a different loss (e.g. Combined Loss with a lower IoU weight).

---

## 8. Reference

Implementation: `src/models/losses.py` — class `FocalLoss`.

Formula in code:

- `error = |pred - target|`
- `error_normalized = clamp(error / max_error, 0, 1)`
- `modulating_factor = error_normalized ** gamma`
- `alpha_mask`: lobe (target ≥ threshold) → `alpha`, else `(1 - alpha)`
- `focal_loss = alpha_mask * modulating_factor * (pred - target)²`, then mean over pixels.
