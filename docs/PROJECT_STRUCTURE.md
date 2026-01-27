# Proposed Project Structure

```
qaanaaq_lobes_detection/
│
├── data/                          # All data files (gitignored)
│   ├── raw/                       # Original, unprocessed data
│   │   ├── raster/                # Raw raster files (GeoTIFF, etc.)
│   │   │   ├── imagery/
│   │   │   └── dem/
│   │   └── vector/                # Raw vector files (shapefiles, GeoJSON, etc.)
│   │       ├── annotations/
│   │       └── reference/
│   │
│   ├── processed/                 # Processed/intermediate data
│   │   ├── raster/                # Processed rasters (cropped, normalized, etc.)
│   │   ├── vector/                # Processed vectors (cleaned, reprojected, etc.)
│   │   └── tiles/                 # Tiled data for CNN training
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   ├── outputs/                   # Model outputs and predictions
│   │   ├── predictions/           # Model predictions (rasters/vectors)
│   │   ├── visualizations/        # Maps, plots, overlays
│   │   └── metrics/               # Evaluation metrics and reports
│   │
│   └── models/                    # Saved model files
│       ├── checkpoints/           # Training checkpoints
│       ├── final/                 # Final trained models
│       └── pretrained/            # Pretrained models (if any)
│
├── qgis_project/                  # QGIS project files (existing)
│   ├── *.qgz                      # QGIS project files
│   └── styles/                    # QGIS layer styles (.qml files)
│
├── src/                           # Source code (following .cursorrules)
│   ├── data_processing/           # Data processing modules
│   │   ├── raster_utils.py       # Raster operations
│   │   ├── vector_utils.py       # Vector operations
│   │   ├── geospatial_ops.py     # Coordinate transforms, reprojection
│   │   └── tiling.py             # Tiling for CNN input
│   │
│   ├── preprocessing/             # Preprocessing pipeline
│   │   ├── normalization.py      # Data normalization
│   │   ├── augmentation.py       # Data augmentation
│   │   └── validation.py         # Data validation
│   │
│   ├── models/                    # CNN model definitions
│   │   ├── architectures.py      # Model architectures
│   │   └── losses.py             # Loss functions
│   │
│   ├── training/                  # Training pipeline
│   │   ├── trainer.py            # Training logic
│   │   ├── dataloader.py         # Data loading for training
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── inference/                 # Inference pipeline
│   │   ├── predictor.py          # Prediction logic
│   │   └── postprocessing.py     # Post-process predictions
│   │
│   ├── evaluation/                # Model evaluation
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── visualizer.py         # Visualization utilities
│   │
│   └── utils/                     # General utilities
│       ├── config_loader.py      # Config file loading
│       ├── logging_utils.py      # Logging setup
│       └── path_utils.py         # Path handling
│
├── configs/                       # Configuration files (YAML)
│   ├── data_config.yaml          # Data paths and settings
│   ├── model_config.yaml         # Model architecture config
│   ├── training_config.yaml      # Training hyperparameters
│   └── inference_config.yaml     # Inference settings
│
├── scripts/                       # Thin orchestration scripts
│   ├── preprocess_data.py        # Run preprocessing pipeline
│   ├── train_model.py            # Train CNN model
│   ├── run_inference.py          # Run inference on new data
│   └── evaluate_model.py        # Evaluate trained model
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── exploration/              # Data exploration
│   ├── experiments/              # Model experiments
│   └── visualization/            # Visualization notebooks
│
├── tests/                         # Unit and integration tests
│   ├── unit/                     # Unit tests
│   │   ├── test_raster_utils.py
│   │   ├── test_vector_utils.py
│   │   └── ...
│   └── integration/              # Integration tests
│       └── test_pipeline.py
│
├── personal/                      # Temporary/experimental scripts (gitignored)
│
├── docs/                          # Documentation
│   ├── data_sources.md           # Data documentation
│   ├── pipeline.md               # Pipeline documentation
│   └── api.md                    # API documentation
│
├── .gitignore
├── .cursorrules.md               # Existing rules file
├── pyproject.toml                # Poetry dependencies
├── README.md                     # Project overview
└── PROJECT_STRUCTURE.md          # This file
```

## Key Design Decisions

1. **Data Organization**: Clear separation of raw, processed, and output data following ML best practices
2. **Code Structure**: Follows `.cursorrules` with `src/`, `tests/`, `configs/`, and `scripts/` separation
3. **QGIS Integration**: Dedicated `qgis_project/` folder with styles subfolder
4. **CNN Pipeline**: Separate modules for preprocessing, training, inference, and evaluation
5. **Scalability**: Structure supports growth with clear module boundaries
6. **Config-Driven**: YAML configs for all major components (as per .cursorrules)
7. **Testability**: Dedicated tests folder with unit and integration tests

## Next Steps

1. Create the folder structure
2. Set up `.gitignore` for data directories
3. Initialize Poetry project with dependencies
4. Create initial config file templates

