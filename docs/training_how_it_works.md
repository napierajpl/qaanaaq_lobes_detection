# How Model Training Works

## Key Question: How Does the Model Adjust During Training?

**Answer: The model adjusts ONLY based on the loss function.**

### Training Process (Each Iteration)

1. **Forward Pass**: Model makes predictions
2. **Loss Computation**: `WeightedSmoothL1Loss` calculates error
3. **Backward Pass**: Gradients computed via backpropagation
4. **Optimizer Step**: Weights updated to minimize loss

```python
# This is what happens in train_one_epoch():
outputs = model(features)           # Forward pass
loss = criterion(outputs, targets)  # Loss function (weighted_smooth_l1)
loss.backward()                      # Compute gradients
optimizer.step()                     # Update weights
```

### What the Loss Function Does

The `WeightedSmoothL1Loss`:
- Computes error between predictions and targets
- Applies **5x higher weight** to lobe pixels (value >= 5.0)
- This forces the model to pay more attention to lobes (class imbalance)

**The model has NO knowledge of baseline metrics during training.**

## Baseline Information: For Evaluation Only

Baseline metrics are **NOT used for training**. They are:
- ✅ **Computed once** during tile filtering
- ✅ **Stored in JSON** for reference
- ✅ **Compared during validation** to see if model is improving
- ❌ **NOT used to adjust weights** during training

### Why This Matters

- **Training**: Model learns by minimizing loss (weighted smooth L1)
- **Evaluation**: We compare model performance to baseline to see if it's actually learning

## Baseline Comparison During Training

During validation, we now compare:
- **Model MAE** vs **Baseline MAE** (predict 0 everywhere)
- Shows if model is genuinely improving or just matching baseline

### Example Output

```
Epoch 1 [Val]: loss=0.4523, mae=0.3821, iou=0.0000
  Baseline MAE: 0.3727 | Model MAE: 0.3821 | ✗ WORSE (Δ-0.0094)
```

This tells you:
- Model MAE (0.3821) is worse than baseline (0.3727)
- Model is not learning yet - needs more training

After training improves:
```
Epoch 50 [Val]: loss=0.2156, mae=0.2456, iou=0.1234
  Baseline MAE: 0.3727 | Model MAE: 0.2456 | ✓ BETTER (Δ+0.1271)
```

Now model is genuinely better than baseline!

## Summary

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **What drives updates?** | Loss function only | N/A (no updates) |
| **Baseline used?** | ❌ No | ✅ Yes (comparison) |
| **Purpose** | Learn from data | Measure progress |

**Bottom line**: Baseline info is valuable for **understanding** if you're improving, but the model learns purely from the loss function.
