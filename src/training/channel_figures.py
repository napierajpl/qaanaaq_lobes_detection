"""
Per-channel representative-tile figures (RGB, DEM, Slope, Segmentation, Target, Prediction).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import torch

from src.training.prediction_tiles import _load_rgb_for_display


def _load_raw_feature_channels_for_display(
    features_path: Path, features_base_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(features_path)
    path = (features_base_dir / p) if not p.is_absolute() else p
    empty = (
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
        np.zeros((256, 256), dtype=np.float64),
    )
    if not path.exists():
        return empty
    with rasterio.open(path) as src:
        if src.count < 5:
            return empty
        data = src.read([1, 2, 3, 4, 5])
    r = np.asarray(data[0], dtype=np.float64)
    g = np.asarray(data[1], dtype=np.float64)
    b = np.asarray(data[2], dtype=np.float64)
    dem = np.asarray(data[3], dtype=np.float64)
    slope = np.asarray(data[4], dtype=np.float64)
    return r, g, b, dem, slope


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
    segmentation_base_dir: Path, tile_id: str,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    seg_path = segmentation_base_dir / f"{tile_id}.tif"
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


def create_representative_tiles_channel_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    iou_threshold: float,
    tile_size: int,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    plot_options: Optional[Dict[str, Any]] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> Dict[str, plt.Figure]:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_iou

    if not rep_tiles:
        return {}
    first_path = Path(features_dir) / rep_tiles[0].get("features_path", "").replace("\\", "/")
    if first_path.exists():
        with rasterio.open(first_path) as src:
            h, w = src.height, src.width
            if h == w:
                tile_size = int(h)
    model.eval()
    mode = (target_mode or "proximity").lower()
    vmin, vmax = (0.0, 1.0) if mode == "binary" else (0.0, 20.0)
    dataset = TileDataset(
        rep_tiles, features_dir, targets_dir, normalization_stats,
        tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )
    info_lines = _build_training_params_info_lines(plot_options)
    figures = {}
    n_rows, n_cols = 2, 4
    ax_slot_order = [0, 1, 2, 3, 5, 6, 7]
    for i, tile_info in enumerate(rep_tiles):
        features, target = dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(features_batch)
        pred_np = np.squeeze(pred.cpu().numpy())
        target_np = target.squeeze().numpy()
        fp = tile_info.get("features_path", "").replace("\\", "/")
        tid = tile_info.get("tile_id", f"tile_{i}")
        raw_r, raw_g, raw_b, raw_dem, raw_slope = _load_raw_feature_channels_for_display(
            Path(fp), Path(features_dir)
        )
        rgb = _load_rgb_for_display(Path(fp), Path(features_dir))
        panels = _build_panels(
            rgb, raw_dem, raw_slope, target_np, pred_np,
            features, segmentation_base_dir, slope_stripes_base_dir, tid,
        )
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes_flat = np.atleast_2d(axes).ravel()
        for slot_idx, (panel_title, data, kind) in zip(ax_slot_order, panels):
            ax = axes_flat[slot_idx]
            _render_panel(ax, panel_title, data, kind, vmin, vmax, mode)
            ax.axis("off")
        axes_flat[4].axis("off")
        for j in range(len(panels), len(ax_slot_order)):
            axes_flat[ax_slot_order[j]].axis("off")
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


def _build_panels(
    rgb, raw_dem, raw_slope, target_np, pred_np,
    features, segmentation_base_dir, slope_stripes_base_dir, tid,
) -> list:
    panels = []
    panels.append(("RGB", rgb, "rgb"))
    dem_disp = _channel_to_display(raw_dem)
    mn_dem, mx_dem = float(np.nanmin(raw_dem)), float(np.nanmax(raw_dem))
    panels.append(("DEM", (dem_disp, f"{mn_dem:.1f}–{mx_dem:.1f}"), "gray_label"))
    slope_disp = _channel_to_display(raw_slope)
    mn_sl, mx_sl = float(np.nanmin(raw_slope)), float(np.nanmax(raw_slope))
    panels.append(("Slope", (slope_disp, f"{mn_sl:.1f}–{mx_sl:.1f}"), "gray_label"))
    seg_used = segmentation_base_dir is not None
    seg_result = _load_segmentation_for_display(segmentation_base_dir, tid) if seg_used else None
    if seg_result is not None:
        seg_arr, seg_boundaries, n_seg = seg_result
        panels.append((f"Segmentation ({n_seg} segments)",
                        ("segmentation_contours", _channel_to_display(seg_arr), seg_boundaries),
                        "segmentation_contours"))
    else:
        h, w = raw_dem.shape
        placeholder = np.zeros((h, w), dtype=np.float32)
        seg_title = "Segmentation (disabled)" if not seg_used else "Segmentation (no file)"
        panels.append((seg_title, _channel_to_display(placeholder), "gray"))
    if slope_stripes_base_dir is not None:
        stripe = features[-1].numpy()
        panels.append(("SlopeStripes", _channel_to_display(stripe), "gray"))
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
