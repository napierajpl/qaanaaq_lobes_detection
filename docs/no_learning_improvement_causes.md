# What could cause no improvement of learning?

The model consistently fails to beat the "predict 0" baseline on validation MAE, and validation metrics plateau or worsen while training loss keeps decreasing. Below are **possible causes**, grouped by category, with **what to check** and **what to try**.

---

## 1. Objective / loss mismatch

**Symptom:** Val loss improves in tuning (e.g. 3.43) but val MAE and "improvement over baseline" do not.

- **Loss does not reward "beat baseline".**
  Combined loss (IoU + Weighted Smooth L1) and Focal Loss are not aligned with "minimize MAE" or "do better than predicting 0". The optimizer finds a minimum that can be bad for MAE (e.g. predicting ~6 everywhere under Focal, or improving IoU at the cost of regression).

- **What to check:** Compare, over epochs, val_loss vs val_mae vs improvement_over_baseline. If loss goes down but MAE does not, the objective is misaligned.

- **What to try:**
  - Add an **explicit MAE or mean-matching term** to the loss (e.g. `loss + λ * MAE(pred, target)` or `loss + λ * |pred.mean() - target.mean()|`) so the objective directly rewards beating the baseline.
  - Use **val MAE** (or improvement over baseline) as the **metric for early stopping and model selection**, not val loss.
  - Try a **simpler loss** (e.g. plain Smooth L1 or MSE) for a few runs to see if the model can at least match the global mean; then reintroduce IoU/focal with a small weight.

---

## 2. Data / labels

**Symptom:** Model cannot learn a useful mapping because the signal is weak, noisy, or inconsistent.

- **Weak or inconsistent labels.**
  Proximity maps might be derived from vector lobes that are incomplete, misregistered, or at a different scale. If "true" lobe locations are wrong or ambiguous, the model cannot learn them.

- **Train–val distribution shift.**
  Val tiles might come from different areas, seasons, or sensors; the model fits train statistics and fails on val.

- **Alignment between inputs and targets.**
  We verified one run (tile 20707): same tile record for RGB and target, so no bug there. But if *upstream* tiling or rasterization introduced a systematic offset (e.g. pixel-alignment between imagery and vector), all tiles could be shifted; the model would then learn "blob somewhere" but not "blob here".

- **What to check:**
  - Manually inspect a few train and val tiles: do RGB and proximity align? Are lobes clearly visible in the imagery where proximity is high?
  - Compare train vs val tile metadata (e.g. area, source); check if val is harder (e.g. different terrain).
  - Run a **sanity check**: train with a tiny subset where you know labels are correct; if the model still cannot beat baseline, the issue is likely not only data.

- **What to try:**
  - Improve label quality (better vector data, careful rasterization, check CRS and resolution).
  - Ensure same preprocessing and alignment for train and val.
  - If you find a systematic offset, fix it in the pipeline and retrain.

---

## 3. Task difficulty / definition

**Symptom:** The task as defined may be too hard or ill-posed for the current setup.

- **Proximity vs binary.**
  Predicting continuous proximity (0–20) is harder than predicting a binary mask. The model might need to first learn "where is the lobe" (binary) before learning "how far" (proximity).

- **Scale / resolution.**
  Lobes might be small or thin; at 256×256 the model may not have enough context or resolution to localize them well.

- **Threshold choice.**
  IoU and lobe_weight use a threshold (e.g. 5). If most proximity values are 0–4, "lobe" is rare and the loss may underweight or overweight the wrong pixels.

- **What to check:**
  - Distribution of target values (histogram per tile or globally): share of 0, 1–5, 5–20.
  - Whether a **binary** task (e.g. pred ≥ 1 vs target ≥ 1) is easier: train a small model on binarized targets and see if it beats baseline on that metric.

- **What to try:**
  - **Two-stage or auxiliary head:** e.g. predict a binary "lobe present" mask first, then predict proximity only where lobe is present.
  - **Different threshold** for IoU / lobe_weight to match the actual distribution of proximity.
  - **Larger context:** larger patches or a small spatial context window (if compatible with your pipeline).

---

## 4. Architecture / capacity

**Symptom:** Model is either unable to fit the signal or overfits immediately.

- **Encoder frozen too long or forever.**
  If the encoder stays frozen, the decoder may not get features that are discriminative for *your* task (solifluction lobes); pretrained features are generic.

- **Decoder too weak.**
  A small decoder might not be able to go from encoder features to a sharp proximity map.

- **Receptive field.**
  If lobe size is large relative to patch size, the model might see only part of a lobe and learn incomplete patterns; if lobes are small, the model might need higher resolution or different strides.

- **What to check:**
  - Unfreeze schedule: when does the encoder unfreeze? Do val metrics change after unfreezing?
  - Overfitting: if train loss drops fast and val does not improve from epoch 1, capacity may be too high for the data size or regularization too low.

- **What to try:**
  - **Unfreeze encoder earlier** (e.g. epoch 1 or 5) with a small LR and monitor val MAE.
  - **Stronger regularization:** more dropout, weight decay, or fewer epochs with early stopping on val MAE.
  - **Simpler model:** e.g. baseline U-Net without pretrained encoder, to see if the task is learnable at all with a smaller model.

---

## 5. Optimization / training dynamics

**Symptom:** Optimizer gets stuck in a bad minimum or never gets a useful gradient.

- **Learning rate.**
  Too high: unstable or early plateau. Too low: almost no learning. With combined or Focal loss, the "best" constant prediction (e.g. ~6) can be a local minimum; the optimizer may never leave it.

- **Batch size.**
  Large batches can smooth gradients and sometimes hide useful signal; small batches can be noisy. Val MAE often depends on this indirectly (effective LR, normalization).

- **Gradient flow.**
  Focal with high gamma strongly down-weights many pixels, so gradients are tiny and the model barely moves (see run 3c5caa5b analysis).

- **What to check:**
  - Gradient norms or a short run with gradient clipping disabled to see if gradients explode or vanish.
  - Learning rate schedule: is LR reduced too early or too late relative to val MAE?

- **What to try:**
  - **Lower gamma** (e.g. 2 or 1) if using Focal; avoid gamma ≥ 4.
  - **LR sweep** (e.g. 1e-4, 5e-4, 1e-3) with early stopping on val MAE.
  - **Warm-up** (e.g. 5–10 epochs with lower LR) so the model does not commit to a bad constant too early.
  - **Different optimizer** (e.g. AdamW with careful weight decay) to see if optimization landscape improves.

---

## 6. Overfitting and regularization

**Symptom:** Train loss keeps decreasing, val loss and val MAE plateau or get worse (e.g. run ea6479a3: best val at epoch 26, then 574 more epochs of overfitting).

- **Too many epochs.**
  Without early stopping on val MAE, the model keeps fitting the training set and validation degrades.

- **Too little regularization.**
  Dropout, weight decay, or data augmentation may be insufficient for the model size and data size.

- **What to check:**
  - Epoch of best val loss / best val MAE; if it is always in the first 20–40 epochs, long runs are harmful.
  - Train vs val gap: if it grows quickly, overfitting is strong.

- **What to try:**
  - **Early stopping** on val MAE (or improvement over baseline) with patience 15–30; save and report the best checkpoint by that metric.
  - **Increase dropout** (e.g. decoder_dropout 0.3–0.4) and/or **weight decay**.
  - **Data augmentation** (flips, rotation, mild color/contrast) to improve generalization.
  - **Fewer epochs** by default (e.g. 50–80) and treat long runs as optional experiments.

---

## 7. Baseline and metric definition

**Symptom:** "No improvement" might be overstated or misinterpreted.

- **Baseline (predict 0).**
  Val MAE for "predict 0" is ~1.39 (mean of target). If the model predicts a constant c, MAE is minimized at c = median(target); for a skewed distribution, that can differ from the mean. So the "baseline" we use (0) is simple but not necessarily the best constant predictor.

- **MAE vs loss.**
  We care about MAE and "beat baseline", but we optimize a different loss. As long as that is true, improvement in loss need not imply improvement in MAE.

- **What to check:**
  - Compute best constant predictor (median or mean of val targets) and its MAE; compare to "predict 0" and to the model.
  - Report val MAE and improvement over baseline in every run so we optimize the right thing.

- **What to try:**
  - **Optimize or early-stop on val MAE** (or improvement over baseline), and optionally add an MAE term to the loss so training and evaluation are aligned.

---

## Summary table

| Category        | Possible cause                          | Quick check / try |
|----------------|------------------------------------------|-------------------|
| **Objective**  | Loss ≠ MAE / baseline                    | Add MAE term; early-stop on val MAE |
| **Data**       | Labels weak, shift, or misaligned        | Inspect tiles; sanity run on clean subset |
| **Task**       | Proximity too hard; threshold wrong      | Try binary task; check target distribution |
| **Architecture** | Encoder frozen; decoder weak; overfit   | Unfreeze earlier; more dropout; try smaller model |
| **Optimization** | LR, gamma, bad local minimum            | Lower Focal gamma; LR sweep; warm-up |
| **Regularization** | Overfitting; too many epochs           | Early stop on val MAE; more dropout; fewer epochs |
| **Metric**     | Baseline or metric definition            | Best-constant MAE; report improvement over baseline |

---

## Suggested order of investigation

1. **Align objective with the metric:** Add a small MAE (or mean-matching) term to the loss and use val MAE for early stopping. Re-run one of the best configs (e.g. combined loss, resnet50) for 50–80 epochs. If val MAE still does not beat baseline, the issue is likely not only the loss.
2. **Sanity check on data:** Pick 50–100 tiles where you are confident labels are correct; train a small model (e.g. 20 epochs). If it still cannot beat baseline, look at data and task definition.
3. **Simplify the task:** Train on binary targets (e.g. target ≥ 1 → 1, else 0) with a simple loss (BCE or Dice). If the model beats a constant predictor on that, then reintroduce proximity and combined loss gradually.
4. **Unfreeze and regularize:** Unfreeze encoder earlier with low LR; increase dropout/weight decay; early stop on val MAE. Compare to current best run.

This doc can be updated as you confirm or rule out causes (e.g. add "Verified: …" or "Ruled out: …" next to each item).
