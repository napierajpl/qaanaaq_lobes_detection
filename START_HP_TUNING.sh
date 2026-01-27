#!/bin/bash
# Command to start Optuna hyperparameter tuning
# Estimated runtime: ~8 hours for 30 trials

cd "$(dirname "$0")"

echo "Starting Optuna hyperparameter tuning..."
echo "Estimated time: ~8 hours for 30 trials"
echo ""
echo "This will:"
echo "  - Run 30 trials with hyperparameter optimization"
echo "  - Use MedianPruner to stop underperforming trials early"
echo "  - Log all trials to MLflow experiment 'lobe_detection_hp_tuning'"
echo "  - Save best hyperparameters to configs/best_hyperparameters.yaml"
echo ""
echo "Press Ctrl+C to stop"
echo ""

poetry run python scripts/tune_hyperparameters.py \
    --n-trials 30 \
    --dev \
    --pruning

echo ""
echo "Hyperparameter tuning complete!"
echo "Best hyperparameters saved to: configs/best_hyperparameters.yaml"
echo "View results in MLflow UI: poetry run python scripts/start_mlflow_ui.py"
