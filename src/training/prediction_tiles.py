"""
Prediction-vs-target tile visualizations (RGB | Target | Prediction).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch


def get_representative_tile_ids_for_viz(
    viz_config: dict,
    mode: str,
    tile_size: int = 256,
) -> List[Union[int, str]]:
    key_256 = "representative_tile_ids_synthetic_parenthesis_256"
    key_512 = "representative_tile_ids_synthetic_parenthesis_512"
    if mode == "synthetic_parenthesis_256" and key_256 in viz_config:
        return viz_config.get(key_256, [])
    if mode == "synthetic_parenthesis_512" and key_512 in viz_config:
        return viz_config.get(key_512, [])
    if tile_size == 512 and mode == "dev" and "representative_tile_ids_dev_512" in viz_config:
        return viz_config.get("representative_tile_ids_dev_512", [])
    if tile_size == 512 and mode != "dev" and "representative_tile_ids_512" in viz_config:
        return viz_config.get("representative_tile_ids_512", [])
    if mode == "dev" and "representative_tile_ids_dev" in viz_config:
        return viz_config.get("representative_tile_ids_dev", [])
    return viz_config.get("representative_tile_ids", [])


def _tile_id_to_index(tile_id: str) -> Optional[int]:
    parts = tile_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def resolve_representative_tiles(
    all_tiles: List[dict],
    config_ids: List[Union[int, str]],
) -> List[dict]:
    requested_indices = {x for x in config_ids if isinstance(x, int)}
    requested_ids_str = {str(x).strip() for x in config_ids if isinstance(x, str)}
    out = []
    for t in all_tiles:
        tid = t.get("tile_id")
        if tid is None:
            continue
        idx = _tile_id_to_index(tid)
        if idx is not None and idx in requested_indices:
            out.append(t)
        elif tid in requested_ids_str:
            out.append(t)
    return out


def _load_rgb_for_display(features_path: Path, features_base_dir: Path) -> np.ndarray:
    p = Path(features_path)
    path = (features_base_dir / p) if not p.is_absolute() else p
    if not path.exists():
        return np.zeros((256, 256, 3), dtype=np.float32)
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float64)
    if rgb.size == 0:
        return np.zeros((256, 256, 3), dtype=np.float32)
    mx = float(np.max(rgb))
    if mx <= 0:
        pass
    elif mx <= 1.0:
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb.astype(np.float32)


def _add_proximity_scale_and_extrema(
    ax: plt.Axes,
    data: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 20.0,
    colorbar_label: str = "proximity",
) -> None:
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.7, label=colorbar_label)
    flat = np.nan_to_num(data, nan=np.nanmean(data)).ravel()
    if flat.size == 0:
        return
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    min_idx = np.argmin(data)
    max_idx = np.argmax(data)
    min_rc = np.unravel_index(min_idx, data.shape)
    max_rc = np.unravel_index(max_idx, data.shape)
    ax.scatter(min_rc[1], min_rc[0], marker="x", s=100, c="black", linewidths=2, zorder=5)
    ax.scatter(max_rc[1], max_rc[0], marker="x", s=100, c="black", linewidths=2, zorder=5)
    ax.text(min_rc[1], min_rc[0] - 13, f"{min_val:.1f}", color="black", fontsize=8, ha="center", va="top", zorder=6)
    ax.text(max_rc[1], max_rc[0] + 13, f"{max_val:.1f}", color="black", fontsize=8, ha="center", va="bottom", zorder=6)


def _infer_tile_size(features_dir: Path, tiles: List[dict], default: int) -> int:
    if not tiles:
        return default
    first_path = Path(features_dir) / tiles[0].get("features_path", "").replace("\\", "/")
    if first_path.exists():
        with rasterio.open(first_path) as src:
            if src.height == src.width:
                return int(src.height)
    return default


def _create_tile_prediction_figure(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    title: str,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

    tile_size = _infer_tile_size(Path(features_dir), [tile_info], tile_size)
    model.eval()
    mode = (target_mode or "proximity").lower()
    dataset = TileDataset(
        [tile_info], Path(features_dir), Path(targets_dir), normalization_stats,
        tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )
    features, target = dataset[0]
    features_batch = features.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(features_batch)
    pred_np = np.squeeze(pred.cpu().numpy())
    target_np = target.squeeze().numpy()
    pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)
    mae = compute_mae(pred_t, target_t)
    rmse = compute_rmse(pred_t, target_t)
    iou = compute_iou(pred_t, target_t, threshold=iou_threshold)
    fp = tile_info.get("features_path", "")
    rgb = _load_rgb_for_display(Path(fp.replace("\\", "/")), Path(features_dir))
    if mode == "binary":
        vmin, vmax = 0.0, 1.0
        colorbar_label = "lobe (0-1)"
        target_title = "Target (0/1)"
    else:
        vmin, vmax = 0.0, 20.0
        colorbar_label = "proximity"
        target_title = "Proximity (target)"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].set_title(target_title)
    _add_proximity_scale_and_extrema(axes[1], target_np, vmin, vmax, colorbar_label)
    axes[1].axis("off")
    axes[2].set_title("Prediction")
    _add_proximity_scale_and_extrema(axes[2], pred_np, vmin, vmax, colorbar_label)
    axes[2].axis("off")
    subtitle = f"{title}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  IoU: {iou:.4f}"
    if mode == "binary" and np.max(target_np) <= 0:
        subtitle += "  (background tile — target all 0)"
    fig.suptitle(subtitle, fontsize=10)
    plt.tight_layout()
    return fig


def show_tile_prediction(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    title: str,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    return _create_tile_prediction_figure(
        model, tile_info, features_dir, targets_dir, normalization_stats,
        device, tile_size, iou_threshold, title,
        target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )


def show_best_predicted_tile(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    loss_value: float,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    tid = tile_info.get("tile_id", "?")
    title = f"Lowest-loss tile  |  {tid}  |  loss: {loss_value:.6f}"
    return show_tile_prediction(
        model, tile_info, features_dir, targets_dir, normalization_stats,
        device, tile_size, iou_threshold, title,
        target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )


def show_highest_iou_tile(
    model: torch.nn.Module,
    tile_info: dict,
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    tile_size: int,
    iou_threshold: float,
    iou_value: float,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    tile_loss: Optional[float] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> plt.Figure:
    tid = tile_info.get("tile_id", "?")
    title = f"Highest IoU tile  |  {tid}  |  IoU: {iou_value:.4f}"
    if tile_loss is not None:
        title += f"  |  loss: {tile_loss:.6f}"
    return show_tile_prediction(
        model, tile_info, features_dir, targets_dir, normalization_stats,
        device, tile_size, iou_threshold, title,
        target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )


def create_prediction_tile_figures(
    model: torch.nn.Module,
    rep_tiles: List[dict],
    features_dir: Path,
    targets_dir: Path,
    normalization_stats: dict,
    device: torch.device,
    iou_threshold: float = 5.0,
    tile_size: int = 256,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> Dict[str, plt.Figure]:
    from src.training.dataloader import TileDataset
    from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou

    tile_size = _infer_tile_size(features_dir, rep_tiles, tile_size)
    model.eval()
    mode = (target_mode or "proximity").lower()
    if mode == "binary":
        vmin, vmax = 0.0, 1.0
        target_title = "Target (0/1)"
    else:
        vmin, vmax = 0.0, 20.0
        target_title = "Proximity (target)"
    dataset = TileDataset(
        rep_tiles, features_dir, targets_dir, normalization_stats,
        tile_size=tile_size, target_mode=target_mode, binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir, slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb, use_dem=use_dem, use_slope=use_slope,
    )
    figures = {}
    for i, tile_info in enumerate(rep_tiles):
        features, target = dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(features_batch)
        pred_cpu = pred.squeeze(0).cpu()
        target_cpu = target.unsqueeze(0)
        pred_np = np.squeeze(pred_cpu.numpy())
        target_np = target.squeeze().numpy()
        mae = compute_mae(pred_cpu, target_cpu)
        rmse = compute_rmse(pred_cpu, target_cpu)
        iou = compute_iou(pred_cpu, target_cpu, threshold=iou_threshold)
        fp = tile_info.get("features_path", "")
        rgb = _load_rgb_for_display(Path(fp.replace("\\", "/")), Path(features_dir))
        tid = tile_info.get("tile_id", f"tile_{i}")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        im1 = axes[1].imshow(target_np, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im1, ax=axes[1], shrink=0.7, label="Target" if mode == "binary" else "0–20")
        axes[1].set_title(target_title)
        axes[1].axis("off")
        im2 = axes[2].imshow(pred_np, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im2, ax=axes[2], shrink=0.7, label="Pred" if mode == "binary" else "0–20")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        title = f"Tile: {tid}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  IoU: {iou:.4f}"
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        figures[tid] = fig
    return figures
