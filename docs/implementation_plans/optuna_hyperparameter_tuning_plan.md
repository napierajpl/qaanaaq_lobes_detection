# Optuna Hyperparameter Tuning Implementation Plan

**Date**: January 26, 2026
**Status**: Planning Phase

## Overview

Integrate Optuna for automated hyperparameter optimization to systematically explore the hyperparameter space and find optimal configurations for lobe detection model training.

---

## Hyperparameter Analysis

### High-Impact Hyperparameters (Recommended for Tuning)

#### 1. **Learning Rate** ⭐⭐⭐
- **Current**: `0.01`
- **Type**: Continuous (log scale)
- **Range**: `[1e-4, 1e-1]` (0.0001 to 0.1)
- **Distribution**: `suggest_loguniform`
- **Rationale**: Most critical hyperparameter, affects convergence speed and final performance
- **Notes**: Wide range to explore different learning regimes

#### 2. **Batch Size** ⭐⭐⭐
- **Current**: `16`
- **Type**: Categorical (power of 2)
- **Options**: `[8, 16, 32, 64]`
- **Distribution**: `suggest_categorical`
- **Rationale**: Affects gradient stability, training speed, and generalization
- **Notes**: Limited by GPU memory; larger batches may improve stability

#### 3. **Weight Decay** ⭐⭐
- **Current**: `0.001`
- **Type**: Continuous (log scale)
- **Range**: `[1e-5, 1e-2]` (0.00001 to 0.01)
- **Distribution**: `suggest_loguniform`
- **Rationale**: Important for regularization and preventing overfitting
- **Notes**: Current value (0.001) is in middle of range

#### 4. **Focal Loss Alpha** ⭐⭐⭐
- **Current**: `0.75`
- **Type**: Continuous
- **Range**: `[0.25, 0.95]`
- **Distribution**: `suggest_uniform`
- **Rationale**: Critical for handling class imbalance (93.5% background vs 6.5% lobes)
- **Notes**: Higher values favor lobe pixels; current 0.75 is already high

#### 5. **Focal Loss Gamma** ⭐⭐
- **Current**: `2.0`
- **Type**: Continuous
- **Range**: `[1.0, 4.0]`
- **Distribution**: `suggest_uniform`
- **Rationale**: Controls focus on hard examples
- **Notes**: Standard value is 2.0; higher = more focus on hard examples

#### 6. **Decoder Dropout** ⭐⭐
- **Current**: `0.2`
- **Type**: Continuous
- **Range**: `[0.0, 0.5]`
- **Distribution**: `suggest_uniform`
- **Rationale**: Regularization for decoder; encoder has no dropout
- **Notes**: Current 0.2 is moderate; may need higher for overfitting

#### 7. **Learning Rate Scheduler Patience** ⭐
- **Current**: `10`
- **Type**: Integer
- **Range**: `[5, 20]`
- **Distribution**: `suggest_int`
- **Rationale**: How long to wait before reducing LR
- **Notes**: Lower = more aggressive LR reduction

#### 8. **Learning Rate Scheduler Factor** ⭐
- **Current**: `0.5`
- **Type**: Continuous
- **Range**: `[0.1, 0.9]`
- **Distribution**: `suggest_uniform`
- **Rationale**: How much to reduce LR when plateau detected
- **Notes**: Lower = more aggressive reduction

#### 9. **Max Gradient Norm** ⭐
- **Current**: `1.0`
- **Type**: Continuous (log scale)
- **Range**: `[0.1, 10.0]`
- **Distribution**: `suggest_loguniform`
- **Rationale**: Gradient clipping for training stability
- **Notes**: Prevents exploding gradients

#### 10. **Encoder Unfreeze Epoch** ⭐
- **Current**: `10`
- **Type**: Integer
- **Range**: `[0, 50]`
- **Distribution**: `suggest_int`
- **Rationale**: When to unfreeze encoder for fine-tuning
- **Notes**: 0 = never unfreeze; higher = longer frozen period

### Medium-Impact Hyperparameters (Optional)

#### 11. **Loss Function** (Categorical Choice)
- **Current**: `"focal"`
- **Type**: Categorical
- **Options**: `["focal", "combined", "weighted_smooth_l1"]`
- **Distribution**: `suggest_categorical`
- **Rationale**: Different loss functions may work better for different architectures
- **Notes**: Only tune if you want to explore loss functions; focal is already best for imbalance

#### 12. **Encoder Name** (Categorical Choice)
- **Current**: `"resnet50"`
- **Type**: Categorical
- **Options**: `["resnet50", "resnet152", "swin_v2_base", "swin_v2_tiny"]`
- **Distribution**: `suggest_categorical`
- **Rationale**: Different encoders have different capacities and characteristics
- **Notes**: Significant impact but also significant training time difference

### Low-Impact / Should NOT Tune

- **Architecture**: Keep `"satlaspretrain_unet"` (already best)
- **Number of Epochs**: Keep high (300), early stopping handles it
- **Data Splits**: Fixed (train/val/test)
- **Early Stopping Patience**: Keep high (40) to allow convergence
- **Normalization Flags**: Always `true` (standard practice)
- **IoU Threshold**: Task definition, not hyperparameter
- **In/Out Channels**: Fixed by data (5 channels in, 1 channel out)

---

## Implementation Plan

### Phase 1: Setup & Integration

#### 1.1 Add Optuna Dependency
- Add `optuna` to `pyproject.toml` (dev dependencies or main)
- Version: `^3.5.0` (latest stable)

#### 1.2 Create Optuna Study Script
- **File**: `scripts/tune_hyperparameters.py`
- **Purpose**: Main script for running hyperparameter optimization
- **Features**:
  - Load base config
  - Define search space
  - Create Optuna study
  - Run optimization trials
  - Save best parameters

#### 1.3 Refactor Training Script
- **Option A**: Extract training logic into a function that accepts hyperparameters
- **Option B**: Create a wrapper function that Optuna can call
- **Recommendation**: Option A - extract `train_model(config, trial=None)` function

### Phase 2: Objective Function Design

#### 2.1 Objective Function Signature
```python
def objective(trial: optuna.Trial, base_config: dict) -> float:
    """
    Optuna objective function.

    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary

    Returns:
        Validation loss (to minimize)
    """
```

#### 2.2 Hyperparameter Suggestions
- Use Optuna's suggest methods based on parameter type
- Log all suggested parameters to MLflow
- Use trial number for run naming

#### 2.3 Early Stopping Integration
- Use Optuna's `TrialPruner` for early stopping
- Prune trials that are clearly underperforming
- Use `MedianPruner` or `SuccessiveHalvingPruner`

### Phase 3: MLflow Integration

#### 3.1 Trial Tracking
- Each trial = one MLflow run
- Log all hyperparameters to MLflow
- Log final validation metrics
- Tag runs with `optuna_trial` and `study_name`

#### 3.2 Study Tracking
- Create separate MLflow experiment for hyperparameter tuning
- Log best trial parameters
- Log optimization history (parallel coordinates plot, etc.)

### Phase 4: Search Strategy

#### 4.1 Sampler Selection
- **TPESampler** (Tree-structured Parzen Estimator) - Recommended
  - Good for continuous and categorical parameters
  - Handles conditional parameters well
  - Default for Optuna
- **RandomSampler** - For baseline comparison
- **GridSampler** - Only if very few parameters

#### 4.2 Pruning Strategy
- **MedianPruner** - Recommended
  - Prunes bottom 50% of trials at each step
  - Good for early stopping
- **SuccessiveHalvingPruner** - More aggressive
- **HyperbandPruner** - For resource-constrained scenarios

#### 4.3 Number of Trials
- **Initial**: 20-30 trials to explore space
- **Full optimization**: 50-100 trials
- **Budget-based**: Stop when improvement plateaus

---

## Proposed Hyperparameter Ranges Summary

| Hyperparameter | Current | Type | Range/Options | Distribution |
|---------------|---------|------|---------------|--------------|
| **learning_rate** | 0.01 | Continuous | [1e-4, 1e-1] | LogUniform |
| **batch_size** | 16 | Categorical | [8, 16, 32, 64] | Categorical |
| **weight_decay** | 0.001 | Continuous | [1e-5, 1e-2] | LogUniform |
| **focal_alpha** | 0.75 | Continuous | [0.25, 0.95] | Uniform |
| **focal_gamma** | 2.0 | Continuous | [1.0, 4.0] | Uniform |
| **decoder_dropout** | 0.2 | Continuous | [0.0, 0.5] | Uniform |
| **lr_scheduler_patience** | 10 | Integer | [5, 20] | Int |
| **lr_scheduler_factor** | 0.5 | Continuous | [0.1, 0.9] | Uniform |
| **max_grad_norm** | 1.0 | Continuous | [0.1, 10.0] | LogUniform |
| **unfreeze_after_epoch** | 10 | Integer | [0, 50] | Int |

---

## Questions for Discussion

### 1. Scope & Priority
- **Q1**: Should we tune all 10 hyperparameters, or focus on top 5-6 most impactful ones?
  - **Recommendation**: Start with top 5 (LR, batch_size, weight_decay, focal_alpha, focal_gamma)
  - **Rationale**: Reduces search space, faster convergence, easier to interpret

### 2. Loss Function & Architecture
- **Q2**: Should we include loss function and encoder name in the search space?
  - **Recommendation**:
    - **Loss function**: No (focal is already best for imbalance)
    - **Encoder name**: Maybe (but significantly increases search space and time)
  - **Alternative**: Run separate studies for each encoder

### 3. Optimization Objective
- **Q3**: What should be the optimization objective?
  - **Option A**: Validation loss (current recommendation)
  - **Option B**: Validation MAE (more interpretable)
  - **Option C**: Validation IoU (task-specific)
  - **Option D**: Combined metric (e.g., weighted MAE + IoU)
  - **Recommendation**: Start with validation loss, can add multi-objective later

### 4. Trial Budget
- **Q4**: How many trials should we run?
  - **Initial exploration**: 20-30 trials
  - **Full optimization**: 50-100 trials
  - **Budget-based**: Stop when no improvement for N trials
  - **Recommendation**: Start with 30 trials, extend if promising

### 5. Pruning Strategy
- **Q5**: Should we use aggressive pruning to save time?
  - **Option A**: MedianPruner (prune bottom 50%)
  - **Option B**: SuccessiveHalvingPruner (more aggressive)
  - **Option C**: No pruning (let all trials complete)
  - **Recommendation**: MedianPruner for initial runs, no pruning for final optimization

### 6. Data Split
- **Q6**: Should we use the same train/val split for all trials, or use cross-validation?
  - **Option A**: Fixed split (faster, current setup)
  - **Option B**: K-fold cross-validation (more robust, slower)
  - **Recommendation**: Start with fixed split, consider CV if needed

### 7. Early Stopping
- **Q7**: Should we use Optuna's pruning in addition to training's early stopping?
  - **Recommendation**: Yes, use both:
    - Training early stopping: patience=40 (allows convergence)
    - Optuna pruning: MedianPruner (stops clearly bad trials early)

### 8. MLflow Integration
- **Q8**: How should we organize MLflow experiments?
  - **Option A**: Separate experiment for each study
  - **Option B**: All trials in one experiment with tags
  - **Recommendation**: Separate experiment (e.g., "lobe_detection_hp_tuning")

### 9. Best Model Selection
- **Q9**: After optimization, should we:
  - **Option A**: Retrain on full train+val with best params
  - **Option B**: Use the best trial's model directly
  - **Recommendation**: Option A (retrain on more data)

### 10. Implementation Approach
- **Q10**: Should we:
  - **Option A**: Create new script `tune_hyperparameters.py` (recommended)
  - **Option B**: Add `--tune` flag to existing `train_model.py`
  - **Recommendation**: Option A (cleaner separation, easier to maintain)

---

## Implementation Steps

1. **Add Optuna dependency** to `pyproject.toml`
2. **Refactor `train_model.py`** to extract training function
3. **Create `scripts/tune_hyperparameters.py`** with objective function
4. **Integrate MLflow** for trial tracking
5. **Add pruning** for early stopping of bad trials
6. **Create config template** for hyperparameter search space
7. **Test with small number of trials** (5-10) to verify setup
8. **Run full optimization** (20-100 trials)
9. **Analyze results** and select best hyperparameters
10. **Retrain final model** with best hyperparameters

---

## Expected Outcomes

- **Improved model performance**: Better validation metrics (MAE, IoU)
- **Systematic exploration**: Understand hyperparameter sensitivity
- **Reproducible results**: All trials logged in MLflow
- **Time savings**: Automated search vs manual tuning
- **Best practices**: Learn optimal hyperparameter combinations

---

## Next Steps

1. **Answer questions** (Q1-Q10) to finalize approach
2. **Create implementation** based on decisions
3. **Run initial trials** to validate setup
4. **Execute full optimization**
5. **Document best hyperparameters** and results

---

**Last Updated**: January 26, 2026
