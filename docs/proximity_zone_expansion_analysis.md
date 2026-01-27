# Proximity Zone Expansion Analysis

## Current Situation

**Current Proximity Map Configuration:**
- `max_value = 10` (lobe center pixels)
- `max_distance = 10` (proximity zone extends 10 pixels)
- Formula: `value = max(0, 10 - distance)`
- Result: Pixels 0-10 pixels from lobes have values 10-0
- IoU threshold: 1.0 (any pixel ≥ 1.0 is "lobe area")

**Problem:**
- Model predictions stuck at ~0.3-0.4 (never crossing 1.0 threshold)
- IoU = 0.0647 (essentially zero)
- Only 6.5% of pixels are lobe/proximity (93.5% background)

---

## Proposed Solution: Expand to 20 Pixels

### Option A: Expand Both (max_value=20, max_distance=20) ⭐ **Recommended**

**Configuration:**
- `max_value = 20` (lobe center pixels)
- `max_distance = 20` (proximity zone extends 20 pixels)
- Formula: `value = max(0, 20 - distance)`
- Result: Pixels 0-20 pixels from lobes have values 20-0

**Impact:**
- **4x larger target area** (π×20² vs π×10² ≈ 4x more pixels)
- **More training examples**: ~4x more positive pixels per lobe
- **Easier threshold crossing**: More pixels need to cross 1.0
- **Better class balance**: Increases positive pixel ratio from 6.5% to ~20-25%

**Benefits:**
1. ✅ **Larger target = easier to hit**: Model has more opportunities to predict ≥ 1.0
2. ✅ **More training data**: 4x more positive examples per lobe
3. ✅ **Better class balance**: Reduces extreme imbalance (93.5% → ~75% background)
4. ✅ **More spatial context**: Model sees larger regions around lobes
5. ✅ **Easier learning**: Larger target area makes task less difficult

**Risks:**
1. ⚠️ **Changes problem definition**: Detecting proximity zones vs actual lobes
2. ⚠️ **Potential false positives**: Larger zones may include non-lobe areas
3. ⚠️ **Value range change**: 0-20 instead of 0-10 (may need loss adjustments)
4. ⚠️ **IoU interpretation**: Threshold=1.0 now includes much larger area

---

### Option B: Expand Distance Only (max_distance=20, max_value=10)

**Configuration:**
- `max_value = 10` (lobe center pixels)
- `max_distance = 20` (proximity zone extends 20 pixels)
- Formula: `value = max(0, 10 - distance)`
- Result: Pixels 0-10 pixels have values 10-0, pixels 11-20 have value 0

**Impact:**
- **No benefit**: Still only 10 pixels have non-zero values
- Pixels 11-20 pixels away still get value 0
- Doesn't help with threshold crossing

**Verdict:** ❌ Not recommended - provides no benefit

---

## Detailed Analysis: Option A (20/20)

### Class Balance Improvement

**Current (10 pixels):**
- Background: ~93.5%
- Lobe/Proximity: ~6.5%

**With 20 pixels (estimated):**
- Background: ~75-80%
- Lobe/Proximity: ~20-25%

**Impact:**
- Reduces extreme imbalance
- More positive examples per batch
- Model sees more lobe patterns during training

### Threshold Crossing Probability

**Current (10 pixels):**
- Target area: ~314 pixels per lobe (π×10²)
- Model needs to predict ≥ 1.0 for these pixels
- Current predictions: ~0.3-0.4 (never crossing)

**With 20 pixels:**
- Target area: ~1,257 pixels per lobe (π×20²)
- **4x more opportunities** to cross threshold
- Even if model predicts ~0.3-0.4, larger area = more chance some pixels cross 1.0

### Training Data Impact

**More Positive Examples:**
- Current: ~6.5% of pixels are positive
- Expanded: ~20-25% of pixels are positive
- **3-4x more positive training examples**

**Better Batch Composition:**
- Current: Most batches are 90%+ background
- Expanded: Batches have 20-25% positive pixels
- Model sees more lobe patterns per batch

---

## Implementation Plan

### Step 1: Update Proximity Map Generation

**File:** `src/data_processing/raster_utils.py`

```python
# Change default values
def __init__(self, max_value: int = 20, max_distance: int = 20):
```

**File:** `scripts/generate_proximity_map.py`

```python
# Update default arguments
parser.add_argument("--max-value", type=int, default=20, ...)
parser.add_argument("--max-distance", type=int, default=20, ...)
```

### Step 2: Regenerate Proximity Maps

**Action Required:**
- Re-run proximity map generation for all data
- This will create new target tiles with 0-20 range
- **Time investment**: ~30-60 minutes (depends on data size)

**Command:**
```bash
poetry run python scripts/generate_proximity_map.py \
  -i data/processed/raster/rasterized_lobes_raw_by_code.tif \
  -o data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif \
  --max-value 20 --max-distance 20
```

### Step 3: Update Data Pipeline

**File:** `scripts/prepare_training_data.py`

Update to use new proximity map file:
```python
# Change from:
"data/processed/raster/rasterized_lobes_raw_by_code_proximity10px.tif"
# To:
"data/processed/raster/rasterized_lobes_raw_by_code_proximity20px.tif"
```

### Step 4: Update Loss Functions (if needed)

**Considerations:**
- Focal Loss: May need to adjust `max_error` from 10.0 to 20.0
- Other losses: Should work fine with 0-20 range
- IoU threshold: Keep at 1.0 (still makes sense)

**File:** `src/models/losses.py` (FocalLoss)

```python
# Update max_error for normalization
max_error = 20.0  # Changed from 10.0
```

### Step 5: Update Documentation

- Update `docs/model_architecture.md` to reflect 0-20 range
- Update config comments
- Note the change in proximity zone size

---

## Expected Outcomes

### Positive Outcomes

1. **IoU Improvement**: 
   - Current: 0.0647
   - Expected: 0.1-0.3 (model can now hit larger target)
   - **Reason**: 4x larger target area = more opportunities

2. **Better Learning**:
   - More positive examples = better feature learning
   - Better class balance = less dominated by background
   - Model should learn lobe patterns faster

3. **Threshold Crossing**:
   - Larger target = higher probability some predictions cross 1.0
   - Even if average prediction is 0.3-0.4, larger area helps

### Potential Issues

1. **False Positives**:
   - Larger proximity zones may include non-lobe areas
   - Need to monitor precision, not just IoU

2. **Value Range**:
   - 0-20 range may require loss function adjustments
   - Normalization may need updates

3. **Problem Definition**:
   - Are we detecting lobes or proximity zones?
   - May need to adjust evaluation metrics

---

## Recommendation

**✅ Proceed with Option A (max_value=20, max_distance=20)**

**Rationale:**
1. **Addresses core problem**: Makes threshold crossing easier
2. **Improves class balance**: Reduces extreme imbalance
3. **More training data**: 4x more positive examples
4. **Low risk**: Can always revert if it doesn't help
5. **Easy to implement**: Simple parameter change

**Implementation Priority:**
- **High**: This is a data-level solution that could break the threshold barrier
- **Low effort**: Just regenerate proximity maps
- **High potential**: Could solve the IoU=0 problem

**Next Steps:**
1. Implement proximity map expansion (20/20)
2. Regenerate all proximity maps
3. Re-run training with Focal Loss
4. Compare results to previous runs
5. If successful, document the change

---

## Alternative: Gradual Expansion

If 20 pixels seems too aggressive, consider:

1. **Try 15 pixels first** (max_value=15, max_distance=15)
   - Moderate expansion
   - Less risk of false positives
   - Still 2.25x larger target area

2. **Then expand to 20** if 15 helps but not enough

This allows incremental testing rather than jumping straight to 20.

---

## Conclusion

Expanding proximity zones from 10 to 20 pixels is a **promising solution** that addresses the threshold crossing problem by:
- Creating a larger, easier target
- Improving class balance
- Providing more training examples
- Making threshold crossing more likely

**Recommendation**: Implement Option A (20/20) and test with Focal Loss. This data-level change could be the key to breaking the IoU=0 barrier.
