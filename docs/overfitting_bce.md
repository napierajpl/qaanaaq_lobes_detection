# Reducing overfitting when using BCE

When training with **binary** targets and **BCE** (binary cross-entropy), the model can overfit to the training set (train loss keeps dropping while val loss flattens or rises). Below are options that work **without changing the loss type** (still BCE).

## Already in your config

- **Dropout:** `decoder_dropout: 0.4` (and `model.dropout` for plain U-Net). Increase to 0.5 if needed.
- **Weight decay:** `weight_decay: 0.001`. Try 0.005 or 0.01 for stronger L2 regularization.
- **Early stopping:** `early_stopping_patience: 50`. Lower (e.g. 30) stops at a more generalizable checkpoint.
- **Frozen encoder:** With `freeze_encoder: true` and `unfreeze_after_epoch: 2000`, the encoder stays frozen for a long time, which limits capacity and can reduce overfitting.

## Options you can add or tune

### 1. Label smoothing (BCE)

Use **BCE with label smoothing**: targets are softened from 0/1 to e.g. 0.1/0.9 so the model does not overconfidently fit exact labels.

- **Config:** `training.bce_label_smoothing: 0.1` (0 = no smoothing, 0.1 is a common value).
- **Effect:** Slightly higher train loss, often better or similar val loss and generalization.

### 2. Data augmentation (on-the-fly)

Your dataloader already supports rotation + contrast/saturation for **extended** tiles (`use_background_and_augmentation` + entries with `"augment": true`). To regularize more:

- **Option A:** Enable `use_background_and_augmentation` and use an extended set with more augmented lobe tiles.
- **Option B:** Add **random flips / rotations** for **all** training tiles each epoch (not only extended set). That would require a small change in the dataset so every `__getitem__` can apply a random flip/rotation with a given probability.

### 3. Train subsample ratio

- **Config:** `data.train_subsample_ratio: 0.8` (or 0.9).
- **Effect:** Each epoch sees a random subset of tiles. Reduces overfitting to the full train set and can act like regularization.

### 4. Learning rate and unfreeze

- **Lower LR:** You already use 0.0000001 (1e-7) in recent runs; going lower can help if the model still overfits.
- **Unfreeze later (or never):** Keep `unfreeze_after_epoch` high so the encoder stays frozen longer; or set it very high so that in practice only the decoder is trained.

### 5. More validation data

- **Larger val split** or **more tiles** so validation metrics are more stable and early stopping picks a better checkpoint.

---

## Summary table

| Lever              | Where              | Example / note                          |
|--------------------|--------------------|-----------------------------------------|
| Label smoothing    | `training.bce_label_smoothing` | 0.1 (optional; implemented)     |
| Decoder dropout    | `model.decoder_dropout`        | 0.4 → 0.5                             |
| Weight decay       | `training.weight_decay`        | 0.001 → 0.005 or 0.01                 |
| Early stopping     | `training.early_stopping_patience` | 50 → 30                          |
| Train subsample    | `data.train_subsample_ratio`   | 1.0 → 0.8 or 0.9                      |
| On-the-fly augment | Dataset / config               | Random flips/rotations for all tiles  |
| Freeze encoder     | `model.encoder.unfreeze_after_epoch` | Keep high or increase             |

All of these are compatible with BCE; no need to switch loss.
