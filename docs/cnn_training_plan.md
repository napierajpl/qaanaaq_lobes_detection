# CNN Training Plan for Lobe Detection

## Task Overview

**Input**: 5-band raster tiles (256×256 pixels)
- RGB imagery: 3 bands (uint8, 0-255)
- DEM: 1 band (float32, elevation in meters)
- Slope: 1 band (float32, degrees or ratio)

**Output**: Single-band proximity map (256×256 pixels)
- Values: 0-10 (uint8)
- 0 = background (no lobes, >10 pixels from lobe)
- 1-9 = proximity decay (distance from lobe)
- 10 = lobe pixels (actual lobe locations)

**Task Type**: Semantic segmentation (pixel-wise regression/classification)

---

## Proposed CNN Architecture: U-Net

### Why U-Net?

1. **Proven for Semantic Segmentation**
   - Industry standard for pixel-wise prediction tasks
   - Originally designed for medical image segmentation
   - Widely adopted in remote sensing and geospatial applications

2. **Encoder-Decoder Architecture**
   - **Encoder**: Extracts multi-scale features (contracting path)
   - **Decoder**: Reconstructs spatial resolution (expanding path)
   - **Skip connections**: Preserve fine-grained spatial details

3. **Handles Multi-Band Input**
   - Can process 5 input channels naturally
   - Flexible input layer accepts any number of bands

4. **Good for Small Objects**
   - Skip connections help preserve small lobe features
   - Multi-scale feature extraction captures context

5. **Efficient Training**
   - Relatively lightweight compared to modern transformers
   - Fast inference suitable for large rasters
   - Can use pretrained encoder backbones (transfer learning)

### Architecture Variants to Consider

#### Option 1: Classic U-Net (Baseline)
- **Encoder**: 4-5 downsampling blocks (conv + maxpool)
- **Decoder**: 4-5 upsampling blocks (upsample + conv)
- **Skip connections**: Concatenate encoder features to decoder
- **Output**: Single channel with regression head (sigmoid/linear)

**Pros**: Simple, interpretable, fast training
**Cons**: May need more data, less feature richness

#### Option 2: U-Net with Pretrained Encoder (Recommended)
- **Encoder**: ResNet34/ResNet50 or EfficientNet backbone (pretrained on ImageNet)
- **Decoder**: Standard U-Net decoder with skip connections
- **Transfer Learning**: Freeze encoder initially, fine-tune later

**Pros**: Better feature extraction, faster convergence, proven performance
**Cons**: Slightly more complex, requires pretrained weights

#### Option 3: Attention U-Net
- **Encoder**: Standard or pretrained
- **Attention gates**: Focus on relevant features
- **Skip connections**: Gated by attention

**Pros**: Better focus on lobe features, handles class imbalance
**Cons**: More parameters, slower training

### Recommended: U-Net with ResNet34 Encoder

**Rationale**:
- Best balance of performance and complexity
- ResNet34 is lightweight but effective
- Pretrained weights available (ImageNet)
- Can fine-tune encoder for geospatial domain
- Standard decoder with skip connections

---

## Model Design Details

### Input Processing
- **Normalization**: 
  - RGB: Normalize to [0, 1] (divide by 255)
  - DEM: Standardize (mean=0, std=1) or min-max normalize
  - Slope: Standardize or min-max normalize
- **Data Type**: Float32 for all inputs

### Output Head
- **Option A: Regression** (Recommended)
  - Single output channel
  - Linear activation (values 0-10)
  - Loss: MSE or Smooth L1
  - Post-process: Round to nearest integer, clip to [0, 10]

- **Option B: Classification**
  - 11 output channels (one per class: 0-10)
  - Softmax activation
  - Loss: Cross-entropy (weighted for class imbalance)
  - Post-process: Argmax to get class

**Recommendation**: Start with **Regression (Option A)**
- Simpler, continuous values match proximity concept
- Easier to interpret
- Can switch to classification if needed

### Loss Function

**Primary**: Smooth L1 Loss (Huber Loss)
- Less sensitive to outliers than MSE
- Good for regression tasks
- Smooth gradient near zero

**Alternative**: MSE Loss
- Standard regression loss
- Simple and effective

**Consider**: Weighted Loss
- Weight lobe pixels (value 10) more heavily
- Address class imbalance (most pixels are 0)

### Metrics

**Training Metrics**:
- Loss (Smooth L1 or MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

**Validation Metrics**:
- Loss, MAE, RMSE
- **IoU (Intersection over Union)** for lobe pixels (threshold: value >= 8)
- **Precision/Recall/F1** for lobe detection (binary: lobe vs non-lobe)
- **Distance Error**: Mean distance between predicted and actual lobe centers

**Visualization**:
- Side-by-side: Input RGB, Ground Truth, Prediction
- Error maps: Absolute difference
- Confusion matrix for lobe/non-lobe classification

---

## MLflow Integration

### Why MLflow?

1. **Experiment Tracking**
   - Log hyperparameters, metrics, and model artifacts
   - Compare multiple model runs side-by-side
   - Track training history and validation curves

2. **Model Versioning**
   - Save trained models with metadata
   - Easy model retrieval for inference
   - Track which model performed best

3. **Reproducibility**
   - Log code version, data paths, and configs
   - Reproduce any experiment exactly
   - Share experiments with team

4. **Artifact Management**
   - Store model checkpoints, predictions, visualizations
   - Organize outputs by experiment
   - Easy access to best models

### MLflow Structure

```
mlruns/
├── 0/                    # Default experiment
│   ├── <run_id>/
│   │   ├── artifacts/
│   │   │   ├── model/     # Saved PyTorch model
│   │   │   ├── configs/   # Training configs
│   │   │   ├── predictions/  # Sample predictions
│   │   │   └── plots/     # Training curves, visualizations
│   │   ├── metrics/       # Metrics (loss, MAE, IoU, etc.)
│   │   ├── params/        # Hyperparameters
│   │   └── tags/          # Run metadata
```

### MLflow Features to Use

1. **Automatic Logging**
   - PyTorch model parameters
   - Training metrics (loss, MAE, RMSE)
   - Validation metrics (IoU, F1, Precision, Recall)
   - Learning rate schedule

2. **Custom Artifacts**
   - Save best model checkpoint
   - Save prediction examples (side-by-side RGB, GT, Pred)
   - Save training curves as images
   - Save confusion matrices

3. **Model Registry** (Optional)
   - Register best models
   - Stage models (Staging, Production)
   - Track model lineage

4. **UI Dashboard**
   - View all experiments in web UI
   - Compare runs visually
   - Filter and search experiments
   - Download models and artifacts

### MLflow Integration Points

- **Training Script**: Log all metrics, params, artifacts
- **Evaluation Script**: Log test metrics, predictions
- **Inference Script**: Load models from MLflow
- **Comparison**: Use MLflow UI to compare model variants

---

## Development Plan

### Phase 1: Foundation (Week 1)

#### 1.1 Project Structure Setup
- [ ] Create `src/models/` directory
- [ ] Create `src/training/` directory
- [ ] Create `src/preprocessing/` directory
- [ ] Create `src/evaluation/` directory
- [ ] Add PyTorch and MLflow dependencies to `pyproject.toml`
- [ ] Create `mlruns/` directory (gitignored) for MLflow experiments
- [ ] Create `src/utils/mlflow_utils.py` for MLflow helpers

#### 1.2 Data Loading Pipeline
- [ ] Create `src/training/dataloader.py`
  - Load feature tiles (5-band) and target tiles (1-band)
  - Pair matching (same tile index)
  - Normalization utilities
  - Data validation (check alignment, dimensions)
- [ ] Create `src/preprocessing/normalization.py`
  - RGB normalization (0-255 → 0-1)
  - DEM/Slope standardization
  - Statistics calculation from training set
- [ ] Create train/val/test split utility
  - Split tiles into sets (e.g., 70/15/15)
  - Ensure no data leakage (tiles from same source image stay together)

#### 1.3 Basic U-Net Implementation
- [ ] Create `src/models/architectures.py`
  - Classic U-Net (no pretrained encoder)
  - Configurable depth and channels
  - Output: single channel regression
- [ ] Create `src/models/losses.py`
  - Smooth L1 Loss
  - MSE Loss
  - Weighted variants

#### 1.4 Training Infrastructure
- [ ] Create `src/training/trainer.py`
  - Training loop
  - Validation loop
  - Checkpoint saving
  - MLflow logging integration
- [ ] Create `src/utils/mlflow_utils.py`
  - MLflow experiment setup
  - Logging helpers (metrics, params, artifacts)
  - Model saving to MLflow
  - Visualization artifact saving
- [ ] Create `configs/training_config.yaml`
  - Hyperparameters (learning rate, batch size, epochs)
  - Data paths
  - MLflow experiment name
  - Model save paths

#### 1.5 Basic Evaluation
- [ ] Create `src/evaluation/metrics.py`
  - MAE, RMSE calculations
  - Basic visualization utilities
- [ ] Create `scripts/train_model.py`
  - CLI script to run training
  - Load config, initialize model, train
  - Start MLflow run, log everything
  - Save model to MLflow

**Deliverable**: Can train a basic U-Net on dev tiles, see loss decreasing, all tracked in MLflow

---

### Phase 2: Improvement (Week 2)

#### 2.1 Pretrained Encoder
- [ ] Update U-Net to use ResNet34 encoder
- [ ] Load pretrained ImageNet weights
- [ ] Add option to freeze/unfreeze encoder
- [ ] Test transfer learning approach

#### 2.2 Data Augmentation
- [ ] Create `src/preprocessing/augmentation.py`
  - Horizontal/vertical flips
  - Rotation (90, 180, 270 degrees)
  - Brightness/contrast adjustments (RGB only)
  - Small translations
  - **Important**: Apply same augmentation to features and targets

#### 2.3 Advanced Training Features
- [ ] Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
- [ ] Early stopping
- [ ] Gradient clipping
- [ ] Mixed precision training (optional, for speed)

#### 2.4 Enhanced Evaluation
- [ ] IoU calculation for lobe detection
- [ ] Precision/Recall/F1 metrics
- [ ] Visualization: prediction overlays on RGB
- [ ] Save prediction examples during validation
- [ ] Log all metrics to MLflow
- [ ] Save visualization artifacts to MLflow

**Deliverable**: Model with pretrained encoder, data augmentation, better metrics, all tracked in MLflow

---

### Phase 3: Optimization (Week 3)

#### 3.1 Hyperparameter Tuning
- [ ] Learning rate search (log each run to MLflow)
- [ ] Batch size optimization
- [ ] Loss function variants (weighted loss)
- [ ] Architecture tweaks (depth, channels)
- [ ] Use MLflow UI to compare all variants
- [ ] Identify best hyperparameters from MLflow metrics

#### 3.2 Advanced Loss Functions
- [ ] Focal Loss (if switching to classification)
- [ ] Dice Loss (for segmentation)
- [ ] Combined losses (e.g., L1 + Dice)

#### 3.3 Model Variants
- [ ] Test different encoder backbones (ResNet50, EfficientNet)
- [ ] Test Attention U-Net
- [ ] Compare performance

#### 3.4 Full Dataset Training
- [ ] Prepare full-size training data (not just dev tiles)
- [ ] Train on full dataset
- [ ] Monitor training time and resource usage

**Deliverable**: Optimized model trained on full dataset

---

### Phase 4: Production (Week 4)

#### 4.1 Inference Pipeline
- [ ] Create `src/inference/predictor.py`
  - Load trained model from MLflow (by run_id or model name)
  - Process full-size rasters (not just tiles)
  - Handle tiling for large inputs
  - Stitch predictions back together
- [ ] Create `scripts/run_inference.py`
  - CLI for inference on new data
  - Option to load model from MLflow
  - Log inference results to MLflow (optional)

#### 4.2 Post-Processing
- [ ] Create `src/inference/postprocessing.py`
  - Convert proximity map to binary mask (threshold)
  - Smooth predictions
  - Remove small artifacts
  - Convert back to vector (optional)

#### 4.3 Model Evaluation on Test Set
- [ ] Final evaluation on held-out test set
- [ ] Generate comprehensive metrics report
- [ ] Visualize predictions on test tiles
- [ ] Log test metrics to MLflow
- [ ] Register best model in MLflow Model Registry
- [ ] Tag production-ready models

#### 4.4 Documentation
- [ ] Document model architecture
- [ ] Document training procedure
- [ ] Document inference usage
- [ ] Create example notebooks

**Deliverable**: Complete pipeline from training to inference

---

## Technical Specifications

### Model Architecture (U-Net with ResNet34 Encoder)

```
Input: (batch_size, 5, 256, 256)
  ↓
Custom Input Layer: Conv(5 → 3)  # Adapt 5-band to 3-channel pretrained encoder
  ↓
ResNet34 Encoder (pretrained, frozen initially):
  - Block 1: 64 channels
  - Block 2: 128 channels
  - Block 3: 256 channels
  - Block 4: 512 channels
  ↓
U-Net Decoder:
  - Upsample + Skip Connection from Block 3 → 256 channels
  - Upsample + Skip Connection from Block 2 → 128 channels
  - Upsample + Skip Connection from Block 1 → 64 channels
  - Final Upsample → 32 channels
  ↓
Output Head: Conv(32 → 1)  # Single channel regression
Output: (batch_size, 1, 256, 256)  # Values 0-10
```

### Training Configuration (Initial)

```yaml
model:
  architecture: "unet_resnet34"
  encoder_pretrained: true
  encoder_frozen: false  # Unfreeze after initial training
  input_channels: 5
  output_channels: 1

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "Adam"
  loss_function: "smooth_l1"
  weight_decay: 0.0001
  
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  augmentation: true
  normalize_rgb: true
  standardize_dem: true
  standardize_slope: true

mlflow:
  experiment_name: "lobe_detection"
  tracking_uri: "file:./mlruns"  # Local file storage
  log_artifacts: true
  log_model: true
  log_predictions: true  # Save sample predictions
  log_plots: true  # Save training curves, confusion matrices
```

---

## Success Criteria

### Minimum Viable Model
- [ ] Loss decreases during training
- [ ] Validation loss tracks training loss (no severe overfitting)
- [ ] MAE < 2.0 on validation set
- [ ] Can detect lobe pixels (IoU > 0.3 for lobe class)

### Good Model
- [ ] MAE < 1.0 on validation set
- [ ] IoU > 0.5 for lobe detection
- [ ] F1 score > 0.6 for lobe/non-lobe classification
- [ ] Predictions visually align with ground truth

### Excellent Model
- [ ] MAE < 0.5 on test set
- [ ] IoU > 0.7 for lobe detection
- [ ] F1 score > 0.8
- [ ] Generalizes to unseen areas

---

## Risk Mitigation

### Potential Issues

1. **Class Imbalance**
   - Most pixels are 0 (background)
   - Solution: Weighted loss, focal loss, or oversample tiles with lobes

2. **Small Dataset**
   - Only 36 dev tiles currently
   - Solution: Data augmentation, transfer learning, generate more tiles

3. **Overfitting**
   - Model memorizes training tiles
   - Solution: Regularization, dropout, early stopping, more data

4. **Misalignment**
   - Features and targets not perfectly aligned
   - Solution: Verify alignment (already done), ensure consistent transforms

5. **Different Scales**
   - RGB (0-255) vs DEM/Slope (different ranges)
   - Solution: Proper normalization/standardization

---

## Next Steps

1. **Immediate**: Set up project structure and basic U-Net
2. **Short-term**: Get baseline training working on dev tiles
3. **Medium-term**: Improve with pretrained encoder and augmentation
4. **Long-term**: Train on full dataset and optimize

---

## Questions to Answer During Development

1. Should we use regression or classification?
2. What loss function works best?
3. How much data augmentation is beneficial?
4. Should encoder be frozen or fine-tuned?
5. What's the optimal learning rate schedule?
6. How to handle edge cases (tiles with no lobes)?

---

## MLflow Usage Examples

### Starting an Experiment

```python
import mlflow
from src.utils.mlflow_utils import setup_mlflow_experiment

# Setup experiment
experiment_name = "lobe_detection"
mlflow.set_experiment(experiment_name)

# Start run
with mlflow.start_run(run_name="unet_resnet34_baseline"):
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 16,
        "architecture": "unet_resnet34"
    })
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_loss, val_mae = validate(...)
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        }, step=epoch)
    
    # Save model
    mlflow.pytorch.log_model(model, "model")
    
    # Save artifacts
    mlflow.log_artifact("predictions.png", "visualizations")
```

### Comparing Models

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
# Compare runs, filter by metrics, download models
```

### Loading Model for Inference

```python
import mlflow.pytorch

# Load model by run_id
model = mlflow.pytorch.load_model("runs:/<run_id>/model")

# Or load from model registry
model = mlflow.pytorch.load_model("models:/lobe_detection/Production")
```

---

## References

- U-Net Paper: Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- ResNet Paper: He et al. (2016) "Deep Residual Learning for Image Recognition"
- Remote Sensing U-Net: Many applications in geospatial domain
- PyTorch U-Net: Multiple implementations available as reference
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- MLflow PyTorch Integration: https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
