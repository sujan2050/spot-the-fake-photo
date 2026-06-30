"""
predict.py
----------
Usage:
    python predict.py some_image.jpg
    python predict.py some_image.jpg --time   # also prints latency

Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)

Approach: 8 hand-engineered signals known to differ between real photos and
screen recaptures:
  1. FFT peak ratio     – moire / sub-pixel grid → sharp periodic spectral spikes
  2. FFT high-freq energy – recaptures carry more mid/high spatial freq content
  3. Highlight ratio    – emissive screens create blown-out glare patches
  4. Sharpness uniformity – flat screen plane → uniform focus; real scenes vary
  5. LBP periodicity    – screen pixel grid → low-entropy repetitive micro-texture
  6. Color cast         – panel color profile imparts a systematic tint
  7. Saturation mean    – double-encode + compression mildly compresses gamut
  8. Edge density       – bezels / UI chrome inflate edge count (weak signal)

Combined with a tiny logistic regression (model.json if trained, else sane defaults).
No GPU. No large weights. Runs in ~15–40 ms on a laptop CPU.
"""

import sys
import time

from PIL import Image

from features import feature_vector
from model import load_model, score


def predict(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    vec, _ = feature_vector(img)
    model = load_model()
    return score(vec, model)


def predict_detailed(image_path: str) -> dict:
    """Returns score + per-feature breakdown + latency. Used by the web demo."""
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    vec, feats = feature_vector(img)
    model = load_model()
    s = score(vec, model)
    latency_ms = (time.perf_counter() - t0) * 1000
    threshold = model.get("threshold", 0.5)
    return {
        "score": s,
        "label": "screen" if s >= threshold else "real",
        "threshold": threshold,
        "latency_ms": latency_ms,
        "features": feats,
        "trained": model.get("trained", False),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--time]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    show_time = "--time" in sys.argv

    if show_time:
        result = predict_detailed(path)
        print(f"{result['score']:.4f}")
        print(f"label    : {result['label']}")
        print(f"latency  : {result['latency_ms']:.1f} ms")
        print(f"threshold: {result['threshold']}")
        print(f"trained  : {result['trained']}")
        print("features :")
        for k, v in result["features"].items():
            print(f"  {k:<28} {v:.4f}")
    else:
        print(predict(path))
