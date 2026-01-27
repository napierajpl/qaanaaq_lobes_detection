# Overfitting Analysis: Train Loss < Validation Loss

## What This Means

**Train Loss < Validation Loss = Overfitting**

The model is:
- ✅ **Memorizing** the training data (low train loss)
- ❌ **Not generalizing** to new data (higher val loss)
- 📉 **Gap increases** over time = getting worse

## Why It's Happening

### 1. **Small Dataset** (Primary Cause)
- **36 tiles total** (25 train, 5 val, 6 test)
- **31M parameters** in model
- **Ratio**: ~800K parameters per training tile!
- Model has enough capacity to memorize all training examples

### 2. **No Dropout** (Missing Regularization)
- Model architecture has **no dropout layers**
- Only BatchNorm for regularization
- Need explicit dropout to prevent memorization

### 3. **Low Weight Decay**
- Current: `0.0001` (very low)
- Not enough L2 regularization
- Model weights can grow too large

### 4. **Small Validation Set**
- Only **5 validation tiles**
- Metrics are noisy
- Hard to detect overfitting early

## Solutions

### Immediate Fixes (High Priority)

#### 1. **Add Dropout to Model** ⭐
Add dropout layers to prevent memorization:
- Dropout rate: 0.2-0.5
- Add after each DoubleConv block
- Or add to decoder path

#### 2. **Increase Weight Decay**
```yaml
weight_decay: 0.001  # Increase from 0.0001 (10x)
```

#### 3. **Data Augmentation** ⭐
Artificially increase dataset size:
- Random rotations (90°, 180°, 270°)
- Horizontal/vertical flips
- Brightness/contrast adjustments
- Small translations

#### 4. **Increase Validation Split**
```yaml
train_split: 0.6  # Reduce from 0.7
val_split: 0.25   # Increase from 0.15
test_split: 0.15
```
More validation data = better overfitting detection

### Medium-term Solutions

#### 5. **Reduce Model Capacity**
- Reduce `base_channels` from 64 → 32 or 48
- Fewer parameters = less capacity to overfit
- Trade-off: might reduce model performance

#### 6. **Use Full Dataset**
- Dev mode: 36 tiles (too small!)
- Production: Full dataset (much larger)
- More data = less overfitting

#### 7. **Transfer Learning**
- Use pretrained encoder (ResNet34)
- Freeze early layers
- Only train decoder + final layers
- Less capacity to overfit

## Recommended Action Plan

### Step 1: Add Dropout (Do This First!)
Modify `src/models/architectures.py` to add dropout:
- Add `nn.Dropout2d(0.2)` after each DoubleConv
- Or add dropout to decoder path only

### Step 2: Increase Weight Decay
```yaml
weight_decay: 0.001  # 10x increase
```

### Step 3: Add Data Augmentation
Create augmentation transforms:
- Random rotations
- Random flips
- Apply during training only

### Step 4: Monitor Gap
Track `train_loss - val_loss`:
- **Small gap (< 0.1)**: Good generalization
- **Large gap (> 0.5)**: Overfitting
- **Gap increasing**: Getting worse

## Expected Results

With dropout + augmentation:
- ✅ Train loss will increase slightly (good!)
- ✅ Val loss will decrease (better generalization)
- ✅ Gap will shrink (less overfitting)
- ✅ Model will generalize better to new data

## Success Criteria

Model is not overfitting when:
- ✅ Train loss ≈ Val loss (gap < 0.2)
- ✅ Gap doesn't increase over epochs
- ✅ Val loss continues decreasing
- ✅ Val MAE beats baseline
