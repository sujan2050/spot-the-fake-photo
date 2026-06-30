"""
train.py
--------
Trains a calibrated logistic regression on your real/ and screen/ folders.
Saves weights + normalization stats to model.json so predict.py uses them.

Usage:
    python train.py --real real/ --screen screen/

After running, model.json holds the fitted weights. predict.py automatically
picks it up -- no changes needed to predict.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from features import feature_vector, FEATURE_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def load_dataset(real_dir: str, screen_dir: str):
    X, y, paths = [], [], []

    for label, folder in [(0, real_dir), (1, screen_dir)]:
        folder = Path(folder)
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}", file=sys.stderr)
            continue
        files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"  {label_name(label)}: {len(files)} images in {folder}")
        for fpath in sorted(files):
            try:
                vec, _ = feature_vector(str(fpath))
                X.append(vec)
                y.append(label)
                paths.append(str(fpath))
            except Exception as e:
                print(f"    SKIP {fpath.name}: {e}", file=sys.stderr)

    return np.array(X), np.array(y), paths


def label_name(y): return "screen" if y == 1 else "real"


def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, lr=0.05, epochs=2000, l2=0.01):
    """Mini gradient-descent logistic regression (no sklearn dependency)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        logit = X @ w + b
        p = sigmoid(logit)
        err = p - y
        grad_w = X.T @ err / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def cross_val_accuracy(X, y, folds=5, lr=0.05, epochs=2000, l2=0.01):
    n = len(y)
    idx = np.random.permutation(n)
    fold_size = n // folds
    accs = []
    for k in range(folds):
        val_idx = idx[k * fold_size: (k + 1) * fold_size]
        train_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xval, yval = X[val_idx], y[val_idx]

        mean = Xtr.mean(0); std = np.where(Xtr.std(0) == 0, 1, Xtr.std(0))
        Xtr_n = (Xtr - mean) / std
        Xval_n = (Xval - mean) / std

        w, b = fit_logistic(Xtr_n, ytr, lr, epochs, l2)
        preds = (sigmoid(Xval_n @ w + b) >= 0.5).astype(int)
        accs.append((preds == yval).mean())
    return np.mean(accs), np.std(accs)


def find_best_threshold(X_norm, y, w, b):
    probs = sigmoid(X_norm @ w + b)
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.2, 0.85, 0.05):
        acc = ((probs >= t).astype(int) == y).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",   default="real/",   help="folder of real photos")
    ap.add_argument("--screen", default="screen/", help="folder of screen photos")
    ap.add_argument("--out",    default="model.json")
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("\n── Loading dataset ─────────────────────────────────")
    X, y, paths = load_dataset(args.real, args.screen)

    if len(X) < 10:
        print("ERROR: need at least 10 images total. Collect more photos.", file=sys.stderr)
        sys.exit(1)

    print(f"  Total: {len(X)} images  ({int(y.sum())} screen, {int((1-y).sum())} real)")

    # Normalise
    mean = X.mean(0)
    std  = np.where(X.std(0) == 0, 1.0, X.std(0))
    X_norm = (X - mean) / std

    # Cross-val estimate
    print("\n── Cross-validation (5-fold) ───────────────────────")
    cv_acc, cv_std = cross_val_accuracy(X_norm, y)
    print(f"  Accuracy: {cv_acc*100:.1f}% ± {cv_std*100:.1f}%")

    # Final fit on all data
    w, b = fit_logistic(X_norm, y)
    threshold, train_acc = find_best_threshold(X_norm, y, w, b)

    print(f"\n── Final model ─────────────────────────────────────")
    print(f"  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Threshold      : {threshold:.2f}")
    print(f"\n  Feature weights:")
    for name, wi in sorted(zip(FEATURE_NAMES, w), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(wi) * 10)
        sign = "+" if wi >= 0 else "-"
        print(f"    {name:<28} {sign}{abs(wi):.3f}  {bar}")

    # Save
    model = {
        "feature_names": FEATURE_NAMES,
        "mean":    mean.tolist(),
        "std":     std.tolist(),
        "weights": w.tolist(),
        "bias":    float(b),
        "threshold": float(threshold),
        "trained": True,
        "cv_accuracy": float(cv_acc),
        "n_real":   int((1 - y).sum()),
        "n_screen": int(y.sum()),
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\n  Saved → {args.out}")
    print("\nDone! Run:  python predict.py your_image.jpg")


if __name__ == "__main__":
    main()
