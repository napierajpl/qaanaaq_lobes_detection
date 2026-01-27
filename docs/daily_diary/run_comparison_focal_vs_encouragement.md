# Run Comparison: Focal Loss vs Encouragement Loss

## Overview

Comparison between two loss function experiments to address the extreme class imbalance problem (93.5% background vs 6.5% lobes).

---

## Run Details

### Run 1: Encouragement Loss - `7caed837`
- **Loss Function**: `encouragement` (weight: 10.0)
- **Epochs**: 45 (out of 500 planned)
- **Date**: January 21, 2026
- **Status**: Previous baseline

### Run 2: Focal Loss - `d1037439` ⭐ **New**
- **Loss Function**: `focal` (alpha: 0.75, gamma: 2.0)
- **Epochs**: 37 (stopped early)
- **Date**: January 22, 2026
- **Status**: First Focal Loss experiment

---

## Metrics Comparison

| Metric | Encouragement Loss | Focal Loss | Change |
|--------|-------------------|------------|--------|
| **IoU (Final)** | 0.0647 | 0.0647 | ❌ **No change** |
| **Val MAE (Final)** | 4.92 | 2.47 | ✅ **50% improvement** |
| **Val Loss (Final)** | 7.54 | 0.32 | ✅ **96% reduction** |
| **Baseline MAE** | 0.329 | 0.329 | Same |
| **Model vs Baseline** | 15x worse | 7.5x worse | ✅ **50% improvement** |
| **Improvement %** | -4.59 | -2.14 | ✅ **Better** |

---

## Key Findings

### ✅ What Improved

1. **Validation MAE**: **50% improvement** (4.92 → 2.47)
   - Model is learning to predict values closer to targets
   - Significant progress in regression accuracy

2. **Loss Values**: **96% reduction** (7.54 → 0.32)
   - Note: Different loss scales make direct comparison difficult
   - Focal Loss produces lower absolute values

3. **Baseline Comparison**: **50% improvement**
   - Model went from 15x worse to 7.5x worse than baseline
   - Still not beating baseline, but getting closer

4. **Training Stability**:
   - Focal Loss shows more stable training (less oscillation)
   - Val MAE fluctuates but trends better than encouragement loss

### ❌ What Didn't Improve

1. **IoU: Completely Stuck** (0.0647 in both runs)
   - Model still not detecting lobes
   - Predictions never cross the 1.0 threshold needed for IoU
   - This is the **critical failure** - model can't identify lobe pixels

2. **Still Worse Than Baseline**
   - Model MAE (2.47) vs Baseline MAE (0.329) = 7.5x worse
   - Naive baseline (predict 0 everywhere) still outperforms the model

---

## Detailed Analysis

### IoU Stagnation

**The Problem:**
- IoU = 0.0647 in both runs (essentially identical)
- IoU requires predictions ≥ 1.0 to be considered "lobe area"
- Model predictions are still too low (likely ~0.3-0.4 range)
- **Focal Loss didn't break the threshold barrier**

**Why This Matters:**
- IoU is the primary metric for lobe detection
- Even if MAE improves, if predictions don't cross threshold, IoU stays at 0
- Model is learning to predict values, but not learning to detect lobes

### MAE Improvement

**The Good News:**
- 50% improvement in MAE suggests the model is learning
- Focal Loss is successfully focusing on hard examples
- Model predictions are getting closer to target values

**The Bad News:**
- Improvement is relative - model is still 7.5x worse than baseline
- Better predictions don't help if they're all below the detection threshold
- Model might be learning to predict "average" values rather than lobe values

### Loss Scale Difference

**Observation:**
- Encouragement Loss: ~7.5 (high scale)
- Focal Loss: ~0.32 (low scale)
- Different loss formulations produce different scales

**Implication:**
- Direct loss comparison is misleading
- Focus should be on metrics (MAE, IoU) not raw loss values
- Both losses are being minimized, but with different scales

---

## Insights

### 1. Focal Loss is Working (Partially)

✅ **Evidence:**
- MAE improved significantly
- Loss is decreasing and stable
- Model is learning better than with encouragement loss

❌ **But:**
- Still not solving the core problem (IoU = 0)
- Model still worse than naive baseline
- Threshold barrier remains unbroken

### 2. The Threshold Problem

**Root Cause:**
- Model predictions are in the 0.3-0.4 range (estimated)
- Need predictions ≥ 1.0 for IoU calculation
- Current loss functions don't provide strong enough incentive to cross threshold

**Why Focal Loss Didn't Fix It:**
- Focal Loss focuses on hard examples (high error)
- But if all predictions are low, even "hard" examples have low absolute error
- Model can minimize loss by predicting ~0.3 everywhere (close to most targets)

### 3. Loss Function Limitations

**Current Situation:**
- Both loss functions are regression-based (predict continuous values)
- Neither explicitly encourages crossing the 1.0 threshold
- Model needs a **detection incentive**, not just regression accuracy

**What's Missing:**
- Binary classification component (lobe vs background)
- Explicit threshold-crossing penalty/reward
- Segmentation-focused loss (Dice/IoU) that directly optimizes detection

---

## Conclusions

### Focal Loss Assessment

**Verdict: ⚠️ Partial Success**

- ✅ **Better than Encouragement Loss**: 50% MAE improvement
- ✅ **Learning is happening**: Model predictions improving
- ❌ **Core problem unsolved**: IoU still 0.0
- ❌ **Still worse than baseline**: 7.5x worse

### What This Tells Us

1. **Loss function choice matters**: Focal Loss performs better than Encouragement Loss
2. **But not enough**: Neither loss breaks the threshold barrier
3. **Regression focus is wrong**: Need detection/segmentation focus, not just regression
4. **Threshold crossing is critical**: Model needs explicit incentive to predict ≥ 1.0

### Next Steps

Based on this analysis, potential solutions:

1. **Combine Focal Loss with IoU Loss**
   - Use Focal Loss for regression component
   - Add IoU Loss to directly optimize detection
   - Weight IoU component heavily to force threshold crossing

2. **Binary Classification Approach**
   - Two-stage: first detect (binary), then regress (proximity)
   - Or: use Dice/Focal Loss for binary lobe detection

3. **Threshold-Specific Loss**
   - Create loss that heavily penalizes predictions < 1.0 when target ≥ 1.0
   - Much stronger than current encouragement mechanism

4. **Architecture Changes**
   - Pretrained encoder (better features)
   - Different output activation (sigmoid + scaling to force higher values)

---

## Recommendation

**Focal Loss is a step in the right direction**, but **not the complete solution**. The model is learning better, but still can't detect lobes.

**Priority**: Focus on **breaking the threshold barrier** rather than just improving regression accuracy. Consider combining Focal Loss with a detection-focused loss (IoU/Dice) or switching to a binary classification approach.
