# Training Recommendations for Lobe Detection

## Current Situation
- **IoU = 0 after 40 epochs**: Model isn't predicting any pixels >= 8.0 threshold
- **Training time**: ~1 hour for 40 epochs (~1.5 min/epoch)
- **Current LR**: 0.001 (Adam optimizer)
- **Current epochs**: 50

## Recommendations

### 1. Learning Rate Strategy

**Option A: Increase Learning Rate (Recommended First)**
- Current: 0.001
- Try: **0.002 or 0.003**
- Rationale: If loss is still decreasing but slowly, higher LR can help model learn faster
- Risk: Too high (e.g., 0.01) can cause instability

**Option B: Learning Rate Scheduler (Better Long-term)**
- Start with 0.001, reduce by factor of 0.5 every 20 epochs
- Or use `ReduceLROnPlateau` - reduce when validation loss plateaus
- This allows faster initial learning, then fine-tuning

**Recommendation**: Try **0.002 first**, then add scheduler if needed.

### 2. Number of Epochs

**For Initial Experiments (Dev Mode)**:
- **50-100 epochs** is reasonable
- At ~1.5 min/epoch: 50 epochs = 75 min, 100 epochs = 2.5 hours
- Good for quick iteration and testing different approaches

**For Production Training**:
- **100-200+ epochs** is common for semantic segmentation
- U-Net models often need 100-150 epochs to converge
- With early stopping, you can train longer without wasting time

**Recommendation**: 
- **Start with 100 epochs** for dev mode
- Add **early stopping** (stop if no improvement for 10-15 epochs)
- This gives you flexibility without wasting time

### 3. Other Critical Improvements

**A. Weighted Loss Function** (High Priority!)
- Current: SmoothL1Loss treats all pixels equally
- Problem: Only 2.1% are lobe pixels - model can ignore them
- Solution: Use `WeightedSmoothL1Loss` with `lobe_weight=5.0-10.0`
- This forces model to pay attention to rare lobe pixels

**B. Learning Rate Scheduler**
- Add `ReduceLROnPlateau` or `CosineAnnealingLR`
- Helps model fine-tune after initial learning

**C. Early Stopping**
- Stop if validation loss doesn't improve for 10-15 epochs
- Prevents overfitting and saves time

**D. Monitor Additional Metrics**
- Track max predicted value per epoch
- If max < 8.0, model isn't learning to predict high values
- Track per-class MAE (background vs lobe)

## Recommended Action Plan

### Phase 1: Quick Fix (Next Run)
1. **Increase LR to 0.002**
2. **Increase epochs to 100**
3. **Add early stopping** (patience=15 epochs)
4. Monitor if max predictions start reaching 8.0+

### Phase 2: Better Training (If Phase 1 doesn't help)
1. **Switch to WeightedSmoothL1Loss** (lobe_weight=5.0)
2. **Add learning rate scheduler** (ReduceLROnPlateau)
3. **Track max prediction value** in metrics

### Phase 3: Fine-tuning
1. Experiment with different lobe weights (5.0, 10.0, 20.0)
2. Try different learning rates (0.001, 0.002, 0.003)
3. Compare results in MLflow

## Expected Timeline

- **50 epochs**: ~1.25 hours (current)
- **100 epochs**: ~2.5 hours (recommended for dev)
- **200 epochs**: ~5 hours (for production)

With early stopping, actual training time will be less if model converges early.

## Success Criteria

Model is learning when:
- ✅ Max predicted value increases over epochs (should reach 8.0+)
- ✅ IoU > 0 (even 0.01 is progress!)
- ✅ Validation loss continues decreasing
- ✅ Per-class MAE for lobes decreases

If after 100 epochs with LR=0.002 and weighted loss, IoU is still 0, consider:
- Architecture changes (deeper U-Net, more channels)
- Different loss function (Dice Loss, Focal Loss)
- Data augmentation to increase lobe pixel diversity
