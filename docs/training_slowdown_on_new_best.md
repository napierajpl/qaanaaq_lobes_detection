# Why "new best val_loss" epochs take longer

When validation loss improves, these run and can add noticeable time before the next epoch:

1. **`save_checkpoint()`**
   Writes full model state dict + optimizer state to disk (e.g. `best_model.pt`). For large models this can be hundreds of MB and take several seconds.

2. **Best-tile figures (when this epoch also has a new best tile by loss or IoU)**
   - **`show_best_predicted_tile()`** – Opens raster, builds `TileDataset`, loads one tile from disk, runs model inference, then calls `_create_tile_prediction_figure()` which **loads the same tile and runs inference again** (redundant), loads RGB for display, builds a 3-panel figure.
   - **`show_highest_iou_tile()`** – Calls `_create_tile_prediction_figure()`: one tile load from disk, one inference, RGB load, figure build.
   - **`mlflow.log_figure()`** for each figure – Writes PNGs to the MLflow artifact store.
   - **`fig.savefig()`** for each figure – Writes PNGs to disk.

So the main cost when you see a new best is: checkpoint save + (if best tile/IoU changed) two heavy visualizations, each with disk I/O and model inference.

**What we do:** Best_predicted_tile and best_iou_tile figures are now created only **once at the end of the run** (after loading the best checkpoint), and logged/saved then. During training we only update the in-memory best-tile and best-IoU info. So a "new best val_loss" epoch now only pays for `save_checkpoint()`; the heavy figure work runs once at the end.
