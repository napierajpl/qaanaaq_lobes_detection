# Training Plateau Analysis

## Current Situation

From the dev run metrics:
- **Early epochs (1-5)**: Very high loss/MAE (outliers)
- **Epochs 6-10**: Quick improvement (loss: 3.5→2.3, MAE: 2.0→0.8)
- **After epoch 10**: **Plateau** - loss ~2.3-2.4, MAE ~0.8-0.9
- **Baseline MAE**: 0.6261
- **Model never beats baseline** (always 0.8-0.9 vs 0.6261)

## Root Causes

### 1. **Learning Rate Too High** (Most Likely)
- Current: **0.005** (very high!)
- Problem: High LR causes:
  - Initial quick learning (epochs 6-10)
  - Then overshooting optimal weights
  - Oscillation around local minimum
  - Can't fine-tune to beat baseline

### 2. **No Learning Rate Scheduling**
- LR stays constant at 0.005
- Need to reduce LR after initial learning
- Allows fine-tuning to beat baseline

### 3. **No Early Stopping**
- Training continues even when not improving
- Wastes time and might cause overfitting

### 4. **Small Dataset**
- Only 36 tiles (25 train, 5 val, 6 test)
- Very small validation set (5 tiles) = noisy metrics
- Might need more data or data augmentation

### 5. **Potential Overfitting**
- Model might be memorizing training data
- Validation loss plateaus while train loss might still decrease

## Recommended Solutions

### Immediate Fixes (Next Run)

1. **Reduce Learning Rate**
   ```yaml
   learning_rate: 0.001  # Start lower, use scheduler
   ```

2. **Add Learning Rate Scheduler**
   ```yaml
   lr_scheduler: "ReduceLROnPlateau"
   lr_scheduler_patience: 10
   lr_scheduler_factor: 0.5
   lr_scheduler_min_lr: 0.0001
   ```

3. **Add Early Stopping**
   ```yaml
   early_stopping_patience: 15
   early_stopping_min_delta: 0.001
   ```

4. **Add Gradient Clipping** (for stability)
   ```yaml
   max_grad_norm: 1.0
   ```

### Medium-term Improvements

5. **Monitor Training vs Validation Loss**
   - If train loss decreases but val loss plateaus → overfitting
   - Solution: Add dropout, reduce model capacity, or more data

6. **Increase Validation Set**
   - Current: 5 tiles (too small!)
   - Try: 20% split instead of 15%
   - Or: Use more dev tiles if available

7. **Data Augmentation**
   - Rotations, flips, brightness adjustments
   - Helps with small dataset

8. **Experiment with Loss Function**
   - Current: `lobe_weight=5.0` might not be enough
   - Try: `lobe_weight=10.0` or `20.0`
   - Or: Try Dice Loss for segmentation

## Action Plan

### Step 1: Fix Learning Rate (Priority 1)
```yaml
learning_rate: 0.001  # Reduce from 0.005
lr_scheduler: "ReduceLROnPlateau"
lr_scheduler_patience: 10
lr_scheduler_factor: 0.5
```

### Step 2: Add Early Stopping (Priority 2)
```yaml
early_stopping_patience: 15
```

### Step 3: Add Gradient Clipping (Priority 3)
```yaml
max_grad_norm: 1.0
```

### Step 4: Monitor Overfitting
- Track train_loss vs val_loss
- If gap increases → overfitting

### Step 5: If Still Not Improving
- Increase `lobe_weight` to 10.0 or 20.0
- Try different architecture (deeper U-Net)
- Add data augmentation
- Use full dataset instead of dev tiles

## Expected Results

With these changes:
- **Initial learning**: Slower but more stable
- **Fine-tuning**: LR scheduler allows reaching better minima
- **Baseline**: Should beat baseline (0.6261) after fine-tuning
- **Stability**: Gradient clipping prevents explosions

## Success Metrics

Model is improving when:
- ✅ Val MAE < Baseline MAE (0.6261)
- ✅ Val loss continues decreasing (not plateauing)
- ✅ IoU > 0 (model predicts some lobe pixels)
- ✅ Train/val loss gap doesn't increase (no overfitting)
