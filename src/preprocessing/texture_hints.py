"""
Texture-derived channels to hint at stone-stripe (lobe) areas:
- Coherence: anisotropy from structure tensor (high where stripes dominate).
- Slope alignment: how well local texture direction aligns with slope (aspect).
Computed from RGB and DEM so no separate raster pipeline is needed.
"""

import numpy as np


def _rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) float or uint -> (H, W) float in [0, 1]."""
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        g = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    else:
        g = np.asarray(rgb, dtype=np.float64).squeeze()
    if g.max() > 1.5:
        g = g / 255.0
    return g.astype(np.float64)


def _sobel_xy(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients; return gx, gy (H, W)."""
    from scipy.ndimage import sobel
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return gx, gy


def _gaussian_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth 2D array with Gaussian."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img.astype(np.float64), sigma=sigma, mode="nearest")


def structure_tensor_coherence_and_orientation(
    rgb: np.ndarray,
    sigma_smooth: float = 1.5,
    sigma_structure: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structure tensor from grayscale of rgb. Returns coherence and stripe direction.

    - coherence: (H,W) in [0,1]; high where there is a dominant direction (stripes).
    - stripe_angle: (H,W) in radians; direction along stripes (perpendicular to gradient).
    """
    gray = _rgb_to_grayscale(rgb)
    gray = _gaussian_smooth(gray, sigma_smooth)
    gx, gy = _sobel_xy(gray)
    j_xx = _gaussian_smooth(gx * gx, sigma_structure)
    j_yy = _gaussian_smooth(gy * gy, sigma_structure)
    j_xy = _gaussian_smooth(gx * gy, sigma_structure)

    trace = j_xx + j_yy
    det = j_xx * j_yy - j_xy * j_xy
    diff = j_xx - j_yy
    sqrt_disc = np.sqrt(np.maximum(diff * diff + 4 * j_xy * j_xy, 0))
    lam1 = (trace + sqrt_disc) * 0.5
    lam2 = (trace - sqrt_disc) * 0.5
    eps = 1e-12
    coherence = np.clip((lam1 - lam2) / (lam1 + lam2 + eps), 0, 1).astype(np.float32)

    # Gradient direction: angle of max change (perpendicular to stripe).
    # atan2(2*J_xy, J_xx - J_yy) gives orientation of eigenvector for λ1 (gradient).
    grad_angle = np.arctan2(2 * j_xy, diff + eps)
    # Stripe direction = gradient direction + π/2 (along the stripe).
    stripe_angle = grad_angle + np.pi / 2
    stripe_angle = stripe_angle.astype(np.float32)
    return coherence, stripe_angle


def aspect_from_dem(dem: np.ndarray) -> np.ndarray:
    """
    Aspect (direction of steepest descent) in radians, [0, 2π) or use atan2 convention.
    Central differences; borders get 0.
    """
    dem = np.asarray(dem, dtype=np.float64)
    dy = np.zeros_like(dem)
    dx = np.zeros_like(dem)
    dy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) * 0.5
    dx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) * 0.5
    # Steepest descent: aspect = atan2(-dy, -dx) -> direction downhill.
    aspect = np.arctan2(-dy, -dx).astype(np.float32)
    return aspect


def slope_alignment(
    stripe_angle: np.ndarray,
    aspect: np.ndarray,
) -> np.ndarray:
    """
    How well stripe direction aligns with slope direction.
    Returns (H,W) in [0, 1]: 1 when aligned, 0 when perpendicular.
    """
    diff = stripe_angle - aspect
    cos_diff = np.cos(diff)
    out = (cos_diff + 1) * 0.5
    return np.clip(out, 0, 1).astype(np.float32)


def compute_texture_hint_channels(
    rgb: np.ndarray,
    dem: np.ndarray,
    sigma_smooth: float = 1.5,
    sigma_structure: float = 2.0,
) -> np.ndarray:
    """
    Returns (2, H, W): [coherence, slope_alignment], float32, values in [0, 1].
    """
    coherence, stripe_angle = structure_tensor_coherence_and_orientation(
        rgb, sigma_smooth=sigma_smooth, sigma_structure=sigma_structure
    )
    aspect = aspect_from_dem(dem)
    alignment = slope_alignment(stripe_angle, aspect)
    return np.stack([coherence, alignment], axis=0)
