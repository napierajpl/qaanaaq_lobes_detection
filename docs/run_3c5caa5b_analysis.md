# Analysis: Run 3c5caa5b47c942c4b3da6d911a3a4bb8

## 1. Metrics: training vs validation

| Metric | Epoch 1 | Epoch 96 | Trend |
|--------|---------|----------|--------|
| **Train loss** | 0.902 | **0.387** | Strong decrease |
| **Val loss** | 0.449 | **0.466** | Flat / slightly worse |
| **Val MAE** | 5.76 | ~5.94 | No improvement (~5.7–6.1 throughout) |

- **Best val loss**: 0.4308 (around epoch 57); validation then oscillates and ends worse.
- **Diagnostics** (from your terminal): prediction mean ~6, target mean ~1.39; **100% of predicted pixels ≥ 1.0** vs **~14%** for target. The model is predicting mid–high values almost everywhere instead of sparse proximity.

**Conclusion**: Training loss improves; validation does not. Val MAE stays much worse than baseline (predict 0). Classic signs of overfitting and/or the model minimizing the loss by predicting a flat ~6 everywhere, which hurts MAE and does not improve localization.

---

## 2. Why does the model predict ~6 instead of the true mean (~1.4)?

Even a “dumb” strategy of predicting the global mean everywhere would give mean prediction ≈ 1.4 and much better MAE. The model does not do that because **the loss (Focal Loss in this run) is not optimized by predicting the true mean**.

### Focal Loss and the “optimal constant”

This run used **Focal Loss** with `focal_alpha ≈ 0.51` and **`focal_gamma ≈ 3.96`**.

- **Alpha**: Lobe pixels (target ≥ `lobe_threshold`, default 5) get weight `alpha`; background get `1 - alpha`. So lobe pixels are weighted slightly more. Underestimating lobes is penalized more than overestimating background.
- **Gamma**: Loss per pixel is `alpha_mask * (error_normalized)^gamma * MSE`, with `error_normalized = |pred - target| / max_error`. So **easy examples (small error) are down-weighted** by `error_normalized^gamma`. With gamma ≈ 4, even “medium” errors get small weight.

**Why the model doesn’t move toward the true mean:**

1. **Flat loss when predicting ~6**
   When the model predicts 6 everywhere:
   - On background (target 0): error = 6 → `error_normalized = 6/20 = 0.3` → modulating factor = `0.3^4 ≈ 0.008`. So the gradient that would pull predictions **down** toward 0 (or 1.4) is scaled by ~0.008 and is very small.
   - The loss landscape is **flat** around “predict 6”: gradients are too weak to push the mean prediction down toward 1.4. So the model stays near 6.

2. **Loss-optimal constant is above the data mean**
   If we restrict to a **constant** prediction c:
   - Background (most pixels): we want c small to match 0, but their weight is `(1 - alpha)` and the modulating factor further reduces their influence when error is “medium.”
   - Lobe pixels: we want c to match 5–20; they are weighted by `alpha` and underestimating them (c too low) is penalized.
   So the constant that minimizes Focal Loss is a **compromise** between “match 0 on background” and “don’t underestimate lobes.” That compromise is **above** the global mean 1.4 (often in the 4–6 range). So the loss itself favors a constant above 1.4; the model is doing a “lazy” optimum.

3. **High gamma amplifies the effect**
   With gamma ≈ 4, any pixel with “medium” error (e.g. pred=6, target=0) contributes little gradient. So the signal that would correct background predictions is weak, and the model has no strong incentive to decrease the mean prediction toward the real mean.

**Summary**: The model predicts ~6 because (1) Focal Loss with high gamma makes gradients small for medium errors, so the loss is flat around 6 and the optimizer doesn’t move the mean down; and (2) the loss-optimal constant is above the data mean (1.4) because lobe pixels are weighted more and underestimating them is penalized. So “predict the true mean” is **not** what minimizes this loss.

### What to try

- **Lower gamma** (e.g. 2 or 1): reduces down-weighting of medium errors so the gradient toward the true mean on background is stronger.
- **Add an explicit mean/MAE term**: e.g. `loss + λ * |pred.mean() - target.mean()|` or `loss + λ * MAE(pred, target)` so the objective explicitly rewards matching the global mean.
- **Revisit alpha**: if alpha is too high, the loss is dominated by lobe pixels and the optimum shifts further above the data mean; lowering alpha gives more weight to background and can pull the mean prediction down.

---

## 3. Tile 20707: why does the predicted shape look shifted?

### Data pipeline check (no alignment bug)

- **Same tile record** is used for:
  - RGB: `_load_rgb_for_display(features_path)` from `tile_info["features_path"]`
  - Target: loaded via `TileDataset` from `tile_info["targets_path"]`
  - Model input: same `TileDataset` → same `features_path` → same 256×256 raster
- So RGB, target, and model input are **one and the same tile**; no mix-up between tiles.
- **Dataloader**: `src.read()` and `src.read(1)` with no crop or offset → features and target are same 256×256 grid.
- **Model**: U-Net uses `padding=1` and, in the SatlasPretrain U-Net, a final `interpolate(..., size=input_size)` so output is exactly input size. So there is no systematic shift from the forward pass.

So the **shift you see is not a data or code bug**: the network really predicts a blob of similar “shape” but in a **different location** than the target.

### Why the model might put a similar shape in the wrong place

1. **Loss and “where”**: The combined loss (IoU + regression) can be reduced by getting some overlap and a “reasonable” value range. If the model learns “there is a lobe-like blob somewhere” but not “exactly here,” it can produce a similar-looking blob in a different spot (e.g. where texture/color looks similar). IoU would still be non-zero but low; MAE stays bad.

2. **Spurious cues**: The model may react to appearance (e.g. color, texture) that correlates with lobes in training but appears in multiple places. On tile_20707 it might be activating more on a similar-looking region (e.g. upper-left) than on the true lobe (lower-middle).

3. **Receptive field / context**: With a large receptive field, the model might be using broad context and “guessing” location from that; small shifts or wrong placement can still give a blob that looks structurally similar.

4. **Optimization**: With predictions clustered around ~6 everywhere, the optimizer may find a local minimum where a slight spatial preference (wrong place) still improves the loss a bit, without ever locking onto the correct position.

So: **same tile, same alignment in code; the shift is the model’s learned behavior**, not a bug in how we load or display data.

---

## 4. Recommendations

1. **Reduce overfitting**: Stronger regularization (e.g. dropout, weight decay), or fewer epochs with early stopping tuned on validation MAE / val loss. Consider monitoring **val MAE** and “improvement over baseline” as primary signals; if val MAE does not improve, treat the run as not useful.
2. **Loss / objective**: Consider putting more weight on localization (e.g. IoU or a spatial loss) or trying a loss that penalizes “wrong place” more (e.g. Dice on thresholded maps, or a loss that compares spatial distributions). The current setup may allow “right shape, wrong place” too much.
3. **Inspect alignment in production**: If you ever change tiling, CRS, or resampling, add a one-off check that feature and target rasters are aligned (e.g. same extent, same grid) for a few tiles.
4. **Per-tile diagnostics**: Keep using prediction-tile figures like tile_20707 to spot systematic shifts or repeated wrong locations; that can guide data or architecture changes.

---

## 5. Short summary

- **Metrics**: Train improves, validation does not; val MAE stays high and worse than baseline. Model is effectively predicting ~6 everywhere.
- **Tile 20707 shift**: RGB, target, and model input all come from the same tile and same grid; there is no alignment bug. The network is predicting a similar-shaped blob in a different place because of how it learned (loss, cues, optimization), not because of misaligned data or a wrong tile.
