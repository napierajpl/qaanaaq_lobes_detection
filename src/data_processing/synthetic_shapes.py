"""Shared helpers for drawing synthetic parenthesis shapes on rasters."""
from pathlib import Path
from typing import Optional

import numpy as np


def make_parenthesis_mask(
    char: str,
    height_px: int,
    font_path: Optional[Path] = None,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("PIL is required. Install: pip install Pillow")

    font_size = height_px
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(str(font_path), font_size)
    else:
        for name in (
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                font = ImageFont.truetype(name, font_size)
                break
            except OSError:
                continue
        else:
            font = ImageFont.load_default()

    img = Image.new("L", (font_size * 2, font_size * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, fill=255, font=font)
    arr = np.array(img)
    mask = (arr > 0).astype(np.uint8)
    r, c = np.where(mask)
    if r.size == 0:
        return np.zeros((height_px, height_px), dtype=np.uint8)
    rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()
    return mask[rmin : rmax + 1, cmin : cmax + 1]


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate

    rotated = rotate(
        mask.astype(float), angle_deg, order=0, reshape=True, mode="constant", cval=0
    )
    return (rotated > 0.5).astype(np.uint8)


def place_one_shape(
    rgb: np.ndarray,
    target: np.ndarray,
    center_row: int,
    center_col: int,
    shape_mask: np.ndarray,
    target_value: int = 20,
) -> None:
    """Burn one shape into rgb (black) and target (target_value inside, keep existing outside)."""
    mh, mw = shape_mask.shape
    top = center_row - mh // 2
    left = center_col - mw // 2
    if top < 0 or left < 0:
        return
    if top + mh > rgb.shape[1] or left + mw > rgb.shape[2]:
        return
    roi_r = slice(top, top + mh)
    roi_c = slice(left, left + mw)
    where = shape_mask > 0
    rgb[0:3, roi_r, roi_c][:, where] = 0
    target[roi_r, roi_c] = np.where(where, target_value, target[roi_r, roi_c])
