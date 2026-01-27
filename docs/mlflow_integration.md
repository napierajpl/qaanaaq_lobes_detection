# MLflow Integration Guide

## Overview

MLflow is integrated into the project to track experiments, compare models, and manage artifacts. This makes it easy to:
- Compare different model architectures and hyperparameters
- Track training metrics over time
- Save and retrieve trained models
- Visualize results in a web UI

## Setup

### Installation

MLflow is already included in `pyproject.toml`. Install dependencies:

```bash
poetry install
```

### Directory Structure

```
mlruns/                    # MLflow tracking (gitignored)
├── 0/                    # Default experiment
│   └── <run_id>/         # Individual training runs
│       ├── artifacts/    # Model files, plots, configs
│       ├── metrics/      # Training/validation metrics
│       ├── params/       # Hyperparameters
│       └── tags/         # Run metadata
```

## Usage

### Starting MLflow UI

View all experiments in a web interface:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Basic Workflow

1. **Start Training Run**
   ```python
   import mlflow
   mlflow.set_experiment("lobe_detection")
   
   with mlflow.start_run(run_name="unet_resnet34_v1"):
       # Log parameters
       mlflow.log_params({
           "learning_rate": 0.001,
           "batch_size": 16,
           "architecture": "unet_resnet34"
       })
       
       # Training loop
       for epoch in range(num_epochs):
           train_loss = train_one_epoch(...)
           val_metrics = validate(...)
           
           # Log metrics
           mlflow.log_metrics({
               "train_loss": train_loss,
               "val_loss": val_metrics["loss"],
               "val_mae": val_metrics["mae"]
           }, step=epoch)
       
       # Save model
       mlflow.pytorch.log_model(model, "model")
   ```

2. **Compare Runs**
   - Open MLflow UI
   - Select multiple runs
   - Compare metrics side-by-side
   - Filter by parameters

3. **Load Model for Inference**
   ```python
   import mlflow.pytorch
   
   # By run_id
   model = mlflow.pytorch.load_model("runs:/<run_id>/model")
   
   # From model registry (best model)
   model = mlflow.pytorch.load_model("models:/lobe_detection/Production")
   ```

## What Gets Logged

### Automatic Logging

- **Parameters**: All hyperparameters from config
- **Metrics**: Loss, MAE, RMSE, IoU, F1, Precision, Recall
- **Model**: PyTorch model state dict
- **Code Version**: Git commit hash (if available)

### Custom Artifacts

- **Configs**: Training configuration YAML files
- **Predictions**: Sample prediction images (RGB + GT + Pred)
- **Plots**: Training curves, confusion matrices
- **Checkpoints**: Best model checkpoints

## Experiment Organization

### Experiment Names

- `lobe_detection` - Main experiment for all model variants
- `lobe_detection_ablation` - Ablation studies
- `lobe_detection_hyperopt` - Hyperparameter optimization

### Run Naming Convention

- `unet_resnet34_baseline` - Baseline model
- `unet_resnet34_lr0.001` - Specific hyperparameter variant
- `unet_efficientnet_v1` - Different architecture
- `unet_resnet34_augmented` - With data augmentation

## Model Comparison Workflow

1. **Train Multiple Variants**
   - Different architectures (U-Net, Attention U-Net)
   - Different hyperparameters (learning rates, batch sizes)
   - Different loss functions

2. **View in MLflow UI**
   - Filter runs by architecture
   - Sort by validation MAE or IoU
   - Compare training curves

3. **Select Best Model**
   - Identify best run based on metrics
   - Register model in MLflow Model Registry
   - Tag as "Production" or "Staging"

4. **Use for Inference**
   - Load registered model
   - Run inference on new data
   - Log inference results (optional)

## Best Practices

1. **Always Log Configs**
   - Save training config YAML as artifact
   - Ensures reproducibility

2. **Log Sample Predictions**
   - Save visualizations every N epochs
   - Helps identify training issues early

3. **Use Descriptive Run Names**
   - Include key hyperparameters
   - Makes comparison easier

4. **Tag Important Runs**
   - Tag best models: `mlflow.set_tag("status", "best")`
   - Tag production models: `mlflow.set_tag("stage", "production")`

5. **Regular Cleanup**
   - Archive old experiments
   - Keep only relevant runs

## Integration Points

### Training Script (`scripts/train_model.py`)
- Starts MLflow run
- Logs all metrics and parameters
- Saves model and artifacts

### Evaluation Script (`scripts/evaluate_model.py`)
- Loads model from MLflow
- Logs test metrics
- Saves evaluation visualizations

### Inference Script (`scripts/run_inference.py`)
- Loads model from MLflow (by run_id or name)
- Optionally logs inference results

## Example: Comparing Two Models

```python
# Train Model A
with mlflow.start_run(run_name="unet_resnet34"):
    mlflow.log_param("architecture", "unet_resnet34")
    # ... training ...
    mlflow.log_metric("val_mae", 0.85)
    mlflow.pytorch.log_model(model_a, "model")

# Train Model B
with mlflow.start_run(run_name="unet_efficientnet"):
    mlflow.log_param("architecture", "unet_efficientnet")
    # ... training ...
    mlflow.log_metric("val_mae", 0.92)
    mlflow.pytorch.log_model(model_b, "model")

# Compare in UI:
# mlflow ui
# Open browser, select both runs, compare metrics
```

## Troubleshooting

### MLflow UI Not Starting
- Check if port 5000 is available
- Use `--port` flag: `mlflow ui --port 5001`

### Can't Find Runs
- Check `mlruns/` directory exists
- Verify experiment name matches
- Check tracking URI: `mlflow.get_tracking_uri()`

### Model Loading Fails
- Ensure PyTorch version matches training version
- Check model path is correct
- Verify model was logged successfully

## References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- MLflow PyTorch: https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
