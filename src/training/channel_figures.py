"""Per-channel representative-tile figures — dynamically built from LayerRegistry."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import torch

from src.training.layer_registry import LayerRegistry
from src.training.prediction_tiles import _load_rgb_for_display


def _build_training_params_info_lines(plot_options: Optional[Dict[str, Any]] = None) -> List[str]:
    if not plot_options:
        return []
    info_lines: List[str] = []
    config_summary = plot_options.get("config_summary")
    if config_summary:
        info_lines.extend(s.strip() for s in config_summary.split("|") if s.strip())
    num_train = plot_options.get("num_train_tiles")
    num_val = plot_options.get("num_val_tiles")
    if num_train is not None or num_val is not None:
        tiles = f"tiles: train={num_train}" if num_train is not None else "tiles:"
        if num_val is not None:
            tiles += f" val={num_val}"
        info_lines.append(tiles)
    if plot_options.get("training_start_datetime"):
        info_lines.append(f"started: {plot_options['training_start_datetime']}")
    duration = plot_options.get("training_duration_seconds")
    if duration is not None and duration >= 0:
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        info_lines.append(f"duration: {h}h {m}m")
    wrapped: List[str] = []
    for line in info_lines:
        if len(line) > 100:
            for i in range(0, len(line), 100):
                wrapped.append(line[i : i + 100])
        else:
            wrapped.append(line)
    return wrapped


def _channel_to_display(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    mn, mx = np.min(a), np.max(a)
    if mx - mn > 1e-12:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0, 1).astype(np.float32)


def _segment_boundary_mask(seg: np.ndarray, nodata: float) -> np.ndarray:
    out = np.zeros(seg.shape, dtype=bool)
    valid = seg != nodata
    if not np.any(valid):
        return out
    s = np.asarray(seg, dtype=np.float64)
    out[:-1, :] |= valid[:-1, :] & valid[1:, :] & (s[:-1, :] != s[1:, :])
    out[1:, :]  |= valid[1:, :] & valid[:-1, :] & (s[1:, :] != s[:-1, :])
    out[:, :-1] |= valid[:, :-1] & valid[:, 1:] & (s[:, :-1] != s[:, 1:])
    out[:, 1:]  |= valid[:, 1:] & valid[:, :-1] & (s[:, 1:] != s[:, :-1])
    return out


def _load_segmentation_for_display(
    segmentation_dir: Path, tile_id: str,
) -> Optional[tuple]:
    seg_path = segmentation_dir / f"{tile_id}.tif"
    if not seg_path.exists():
        return None
    with rasterio.open(seg_path) as src:
        seg = src.read(1)
        nd = float(src.nodata if src.nodata is not None else -9999.0)
    seg = np.asarray(seg, dtype=np.float64)
    valid = seg != nd
    out = np.zeros_like(seg, dtype=np.float32)
    n_uniq = 0
    if np.any(valid):
        v = seg[valid].astype(np.int64)
        n_uniq = len(np.unique(v))
        out[valid] = (v % 64) / 63.0
    boundaries = _segment_boundary_mask(seg, nd)
    return (np.clip(out, 0, 1).astype(np.float32), boundaries, n_uniq)


def _load_raw_layer_for_display(
    tile_dir: Path, tile_id: str,
) -> Optional[np.ndarray]:
    path = tile_dir / f"{tile_id}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def _build_panels(
    layer_registry: LayerRegistry,
    tile_id: str,
    target_np: np.ndarray,
    pred_np: np.ndarray,
) -> list:
    panels = []
    for layer in layer_registry.enabled_layers:
        name = layer.spec.name
        display = layer.spec.display
        display_type = display.get("type", "")

        if display_type == "rgb":
            rgb = _load_rgb_for_display(tile_id, layer_registry)
            panels.append((name.upper(), rgb, "rgb"))

        elif display_type == "segmentation":
            seg_result = _load_segmentation_for_display(layer.tile_dir, tile_id)
            if seg_result is not None:
                seg_arr, seg_boundaries, n_seg = seg_result
                panels.append((
                    f"Segmentation ({n_seg} segments)",
                    ("segmentation_contours", _channel_to_display(seg_arr), seg_boundaries),
                    "segmentation_contours",
                ))
            else:
                h, w = target_np.shape
                panels.append(("Segmentation (no file)", np.zeros((h, w), dtype=np.float32), "gray"))

        else:
            raw = _load_raw_layer_for_display(layer.tile_dir, tile_id)
            if raw is None:
                h, w = target_np.shape
                panels.append((name.upper(), np.zeros((h, w), dtype=np.float32), "gray"))
                continue
            disp = _channel_to_display(raw)
            show_range = display.get("show_range", False)
            if show_range:
                mn, mx = float(np.nanmin(raw)), float(np.nanmax(raw))
                panels.append((name.upper(), (disp, f"{mn:.1f}–{mx:.1f}"), "gray_label"))
            else:
                panels.append((name.upper(), disp, "gray"))

    panels.append(("Target", target_np, "viridis"))
    panels.append(("Prediction", pred_np, "viridis"))
    return panels


def _render_panel(ax, title, data, kind, vmin, vmax, mode):
    if kind == "rgb":
        ax.imshow(data)
        ax.set_title(title)
    elif kind == "segmentation_contours":
        _, gray_arr, boundary_mask = data
        ax.imshow(gray_arr, cmap="gray")
        overlay = np.ma.masked_where(~boundary_mask, np.ones_like(boundary_mask, dtype=np.float32))
        white = mcolors.ListedColormap([[1, 1, 1, 1]])
        ax.imshow(overlay, cmap=white, vmin=0, vmax=1, alpha=0.85)
        ax.set_title(title)
    elif kind == "gray":
        im = ax.imshow(data, cmap="gray")
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(title)
    elif kind == "gray_label":
        im_arr, label = data
        im = ax.imshow(im_arr, cmap="gray")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label(label)
        ax.set_title(title)
    else:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im, ax=ax, shrink=0.6, label="0–1" if mode == "binary" else "0–20")
        ax.set_title(title)


def create_representative_tiles_channel_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    targets_dir: Path,
    layer_registry: LayerRegistry,
    device: torch.device,
    iou_threshold: float,
    tile_size: int,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, plt.Figure]:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_iou
    from src.training.prediction_tiles import _infer_tile_size

    if not rep_tiles:
        return {}
    tile_size = _infer_tile_size(layer_registry, rep_tiles, tile_size)
    model.eval()
    mode = (target_mode or "proximity").lower()
    vmin, vmax = (0.0, 1.0) if mode == "binary" else (0.0, 20.0)
    dataset = TileDataset(
        rep_tiles, targets_dir, layer_registry,
        tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
    )
    info_lines = _build_training_params_info_lines(plot_options)
    figures = {}

    n_layers = len(layer_registry.enabled_layers)
    n_panels = n_layers + 2  # layers + target + prediction
    n_cols = min(n_panels, 4)
    n_rows = (n_panels + n_cols - 1) // n_cols

    for i, tile_info in enumerate(rep_tiles):
        features, target = dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(features_batch)
        pred_np = np.squeeze(pred.cpu().numpy())
        target_np = target.squeeze().numpy()
        tid = tile_info.get("tile_id", f"tile_{i}")

        panels = _build_panels(layer_registry, tid, target_np, pred_np)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes_flat = np.atleast_2d(axes).ravel()
        for slot_idx, (panel_title, data, kind) in enumerate(panels):
            if slot_idx >= len(axes_flat):
                break
            ax = axes_flat[slot_idx]
            _render_panel(ax, panel_title, data, kind, vmin, vmax, mode)
            ax.axis("off")
        for j in range(len(panels), len(axes_flat)):
            axes_flat[j].axis("off")

        pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)
        mae = compute_mae(pred_t, target_t)
        iou = compute_iou(pred_t, target_t, threshold=iou_threshold)
        fig_title = f"Representative tile: {tid}  |  MAE: {mae:.4f}  IoU: {iou:.4f}"
        if plot_options and plot_options.get("run_intention"):
            fig_title += "\n" + plot_options["run_intention"]
        fig.suptitle(fig_title, fontsize=11)
        if info_lines:
            fig.subplots_adjust(left=0.02, bottom=0.18, right=0.98, top=0.92)
            fig.text(
                0.02, 0.02, "\n".join(info_lines), fontsize=6, ha="left", va="bottom",
                transform=fig.transFigure, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="lightgray"),
            )
        else:
            plt.tight_layout()
        figures[tid] = fig
    return figures
