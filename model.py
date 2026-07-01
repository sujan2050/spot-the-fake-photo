"""
model.py
--------
Tiny logistic-regression-style scorer on top of the hand-engineered features in
features.py: score = sigmoid(w . standardize(features) + b)

- If model.json exists (written by train.py after you run it on your own real/ +
  screen/ photos), those calibrated weights are used.
- If it doesn't exist yet, predict.py still works out of the box using the
  hand-set DEFAULT weights below -- the brief says training is optional, not required.
  These defaults just encode the *sign* of the reasoning in each feature's docstring
  (e.g. more periodicity -> more screen-like), they are NOT fit on data, and you should
  expect them to be far below the 95% bar. Run train.py to calibrate real weights.
"""
import json
import os
import numpy as np

from features import FEATURE_NAMES

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")

DEFAULT_MODEL = {
    "feature_names": FEATURE_NAMES,
    # Rough, hand-guessed centers/spreads (NOT measured) just so the un-trained default
    # roughly normalizes each feature's scale. Run train.py on real data to replace these
    # with actual statistics + fit weights.
    "mean": [2.0, 0.97, 0.01, 0.5, 0.25, 0.03, 0.15, 0.15],
    "std":  [1.0, 0.03, 0.03, 0.2, 0.15, 0.05, 0.10, 0.10],
    # order matches FEATURE_NAMES: fft_peak_ratio, fft_high_freq_energy, highlight_ratio,
    # sharpness_uniformity, lbp_periodicity, color_cast, saturation_mean, edge_density
    "weights": [0.9, 0.6, 0.5, 0.6, 0.7, 0.5, -0.3, 0.2],
    "bias": -1.6,
    "threshold": 0.45,
    "trained": False,
}


def load_model():
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH) as f:
            return json.load(f)
    return DEFAULT_MODEL


def score(feature_vec: np.ndarray, model: dict) -> float:
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    w = np.array(model["weights"], dtype=np.float64)
    b = float(model["bias"])

    z = (feature_vec - mean) / std
    logit = float(np.dot(w, z) + b)
    logit = np.clip(logit, -60, 60)  # avoid overflow in exp
    return float(1.0 / (1.0 + np.exp(-logit)))