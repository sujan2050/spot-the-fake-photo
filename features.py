"""
features.py
------------
Lightweight, hand-engineered feature extraction for screen-recapture
("photo of a screen", a.k.a. recapture) detection.

Why hand-engineered features instead of a trained CNN?
  - The brief explicitly says training a model is not required, and the point of the
    exercise is figuring out *what actually distinguishes a recapture* -- not training
    on a black box.
  - These features have a physical/optical reason to differ between a REAL scene and a
    RECAPTURE (see each docstring below), so the result is small, fast, and explainable.
  - It runs on a 256-512px thumbnail with plain numpy/OpenCV, no GPU, no deep model
    weights to ship -- friendly to "it will eventually run on a phone".

Each function returns a single float. Higher == more "screen-like" for every feature
(direction is normalized so the whole vector is consistently oriented), except where noted.
"""

import numpy as np
import cv2
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # lets PIL.Image.open() read .heic/.heif files
except ImportError:
    pass  # HEIC files will fail to open with a clear PIL error if this isn't installed

FEATURE_NAMES = [
    "fft_peak_ratio",
    "fft_high_freq_energy",
    "highlight_ratio",
    "sharpness_uniformity",
    "lbp_periodicity",
    "color_cast",
    "saturation_mean",
    "edge_density",
]


def _load_gray_and_rgb(image_path_or_img, max_dim=512):
    """Accepts a path or an already-opened PIL.Image. Downscales for speed."""
    if isinstance(image_path_or_img, Image.Image):
        img = image_path_or_img.convert("RGB")
    else:
        img = Image.open(image_path_or_img).convert("RGB")

    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    rgb = np.array(img).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def fft_features(gray):
    """
    LCD/OLED panels have a regular sub-pixel grid, and a recapture is the camera's own
    sensor grid sampling that grid again. That beating between two regular grids
    ("moire") shows up as sharp, non-radial peaks in the 2D Fourier spectrum on top of
    an otherwise smooth, naturally-decaying spectrum. Real-world scenes very rarely
    produce sharp periodic peaks like this.

    Returns:
        peak_ratio       -- how far the strongest off-center spectral peak rises above
                             the local noise floor (z-score-like). Higher = more periodic
                             structure = more screen-like.
        high_freq_energy -- fraction of spectral energy sitting in the mid/high band
                             (away from the DC/lighting-dominated center).
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    # radial average -> the "expected" smooth baseline spectrum at each radius
    r_flat = r.ravel()
    mag_flat = mag.ravel()
    r_max = int(r_flat.max())
    radial_sum = np.bincount(r_flat, weights=mag_flat, minlength=r_max + 1)
    radial_cnt = np.bincount(r_flat, minlength=r_max + 1)
    radial_mean = radial_sum / np.maximum(radial_cnt, 1)

    expected = radial_mean[r]
    residual = mag - expected  # positive spike = energy above the smooth baseline

    # ignore the very center: dominated by overall exposure/lighting, not texture
    mask = r > (0.04 * min(h, w))
    region = residual[mask]
    if region.size == 0:
        peak_ratio = 0.0
    else:
        peak_ratio = float(np.percentile(region, 99.5) / (region.std() + 1e-6))

    total_energy = float(mag.sum()) + 1e-6
    high_freq_energy = float(mag[mask].sum() / total_energy)

    return peak_ratio, high_freq_energy


def highlight_ratio(rgb):
    """
    Screens are emissive (they're a light source, not a lit object) and re-photographing
    one very often produces glare/reflection hot-spots plus abrupt highlight clipping.
    Returns the fraction of near-blown-out (>245/255) pixels.
    """
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(np.mean(luminance > 245))


def sharpness_uniformity(gray):
    """
    A real scene usually has varying depth (foreground vs. background), so focus/blur
    varies across the frame. A recaptured screen is one flat plane, so sharpness tends
    to be much more *spatially uniform*. We tile the image, measure local Laplacian
    variance (a standard blur/focus proxy) per tile, and score how *consistent* those
    tile values are (low spread == flat plane == more screen-like).
    """
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    h, w = lap.shape
    ny, nx = 6, 6
    tile_vars = []
    for i in range(ny):
        for j in range(nx):
            tile = lap[i * h // ny:(i + 1) * h // ny, j * w // nx:(j + 1) * w // nx]
            if tile.size > 4:
                tile_vars.append(tile.var())
    tile_vars = np.array(tile_vars) if tile_vars else np.array([0.0])
    mean = tile_vars.mean() + 1e-6
    coeff_var = tile_vars.std() / mean
    return float(1.0 / (1.0 + coeff_var))  # higher = more uniform = more screen-like


def lbp_periodicity(gray):
    """
    A cheap local-binary-pattern-style texture descriptor on a downsampled image.
    Screen pixel grids / sub-pixel structure produce a small handful of highly
    repetitive micro-patterns; natural textures (skin, fabric, foliage, walls, wood
    grain...) produce a much more varied pattern histogram. We measure the *entropy*
    of the pattern histogram and invert it: low entropy (few dominant patterns) -> high
    "periodicity" score -> more screen-like.
    """
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.uint8)
    centers = small[1:-1, 1:-1]
    neighbors = [
        small[0:-2, 0:-2], small[0:-2, 1:-1], small[0:-2, 2:],
        small[1:-1, 0:-2],                     small[1:-1, 2:],
        small[2:,   0:-2], small[2:,   1:-1], small[2:,   2:],
    ]
    code = np.zeros_like(centers, dtype=np.uint8)
    for k, n in enumerate(neighbors):
        code |= ((n >= centers).astype(np.uint8) << k)

    hist, _ = np.histogram(code, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    max_entropy = 8.0  # log2(256)
    return float(1.0 - entropy / max_entropy)


def color_cast(rgb):
    """
    Display panels commonly impart a subtle but consistent color tint (cool/blue or
    warm, depending on the panel and its color profile) that a second camera capture
    doesn't fully neutralize. Returns mean per-channel deviation from gray, normalized.
    """
    r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    avg = (r + g + b) / 3 + 1e-6
    return float((abs(r - avg) + abs(g - avg) + abs(b - avg)) / avg)


def saturation_mean(rgb):
    """
    Display + recapture pipelines (panel gamut, compression, a second lens/sensor) tend
    to mildly compress color gamut versus a directly-lit real scene. Mean HSV saturation,
    normalized to [0, 1].
    """
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean() / 255.0)


def edge_density(gray):
    """
    Supporting signal: screens often carry visible bezels, UI chrome, or pixel-grid edges,
    which tends to push edge density up. Weighted lightly in the default model since this
    is the least specific of the features (busy real scenes can also have many edges).
    """
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    return float(np.mean(edges > 0))


def extract_features(image_path_or_img):
    """Returns a dict of {feature_name: value} for one image."""
    rgb, gray = _load_gray_and_rgb(image_path_or_img)
    peak_ratio, high_freq_energy = fft_features(gray)
    return {
        "fft_peak_ratio": peak_ratio,
        "fft_high_freq_energy": high_freq_energy,
        "highlight_ratio": highlight_ratio(rgb),
        "sharpness_uniformity": sharpness_uniformity(gray),
        "lbp_periodicity": lbp_periodicity(gray),
        "color_cast": color_cast(rgb),
        "saturation_mean": saturation_mean(rgb),
        "edge_density": edge_density(gray),
    }


def feature_vector(image_path_or_img):
    """Returns (np.ndarray in FEATURE_NAMES order, dict of named features)."""
    feats = extract_features(image_path_or_img)
    vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
    return vec, feats