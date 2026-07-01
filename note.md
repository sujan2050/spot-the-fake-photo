# Spot the Fake Photo — Submission Note
**Yesireddy Sujan Kumar Reddy**

---

## Approach

The detector uses **classical signal processing**, not a trained neural network.
The key insight: a real photo and a screen-recapture differ in specific,
physically-explainable ways. Eight hand-engineered features capture those
differences:

1. **FFT peak ratio** — LCD/OLED sub-pixel grids beat against the camera
   sensor grid (moiré). This shows up as sharp, non-radial spikes in the 2D
   Fourier spectrum that real scenes almost never produce.
2. **FFT high-freq energy** — Screen recaptures carry anomalously high
   mid/high spatial frequency energy from the double-sampling process.
3. **Highlight ratio** — Emissive displays are a light source; re-photographing
   them creates blown-out glare patches (>245/255 luminance) that naturally-lit
   real scenes rarely produce.
4. **Sharpness uniformity** — A flat screen is one plane, so focus is spatially
   uniform. Real scenes have depth variation (foreground/background), making
   blur non-uniform across tiles.
5. **LBP periodicity** — The screen pixel grid produces a small, repetitive set
   of local micro-texture patterns (low histogram entropy). Natural textures
   (skin, fabric, foliage) produce a much more varied pattern set.
6. **Color cast** — Panel color profiles impart a systematic tint (warm or
   cool) that the second camera does not fully neutralise.
7. **Saturation mean** — Double encode and compression mildly compress the
   color gamut versus a directly-lit real scene.
8. **Edge density** — Bezels and UI chrome push edge count up (weaker signal,
   used as a supporting feature only).

These 8 numbers are combined with a **tiny logistic regression**
(`sigmoid(w · standardize(features) + b)`) fit by hand-written gradient
descent — no sklearn or external ML library needed. The model is 8 weights
plus a bias: the entire `model.json` is under 1 KB.

---

## Dataset

- **Real photos:** 66 images — everyday objects, food, people, outdoor scenes,
  taken with a phone camera indoors and outdoors across varied lighting
  conditions.
- **Screen recaptures:** 74 images — photos of phone and laptop screens
  displaying images, taken at different angles, distances, and lighting, across
  multiple device types to add variety.
- **Format:** JPEG and HEIC (iPhone default); HEIC decoded transparently via
  `pillow-heif` — no manual conversion needed.
- **Total:** 140 images.

---

## How Training Works (Step-by-Step Pipeline)

```
real/          screen/
  ↓               ↓
features.py: extract 8 signal-processing features per image
  ↓
numpy array X (140 × 8),  labels y (0=real, 1=screen)
  ↓
standardize: z = (X − mean) / std   (per-feature, fitted on train split)
  ↓
train.py: gradient descent logistic regression (lr=0.05, 2000 epochs, L2=0.01)
  ↓
5-fold cross-validation → honest accuracy estimate
  ↓
threshold search on training set → best decision boundary
  ↓
model.json: weights, mean, std, bias, threshold, cv_accuracy
  ↓
predict.py: load model.json → extract features → standardize → sigmoid → score
```

Run training with:
```bash
python train.py --real real/ --screen screen/
```

---

## Accuracy

| Metric | Value |
|---|---|
| 5-fold cross-validation accuracy | **75.7% ± 4.7%** |
| Training set accuracy | 81.4% |
| Decision threshold | 0.60 |
| Dataset size | 140 images (66 real, 74 screen) |

The honest number to report is **75.7% cross-val** — this is evaluated on
held-out folds the model did not train on.

This is below the 95% target. The most likely causes:

- **Dataset size** — 140 images with 8 features gives wide confidence
  intervals (±4.7%). The dominant feature weight is `saturation_mean` (+1.288),
  a weak, dataset-dependent signal, which suggests a lighting/environment
  confound in the data rather than a true screen signature.
- **Dataset variety** — all screen shots taken in similar indoor conditions;
  all real shots taken in a different set of conditions. More variety across
  lighting, angles, and screen types would help the model generalise.

---

## Latency

Measured on a laptop CPU (Intel, no GPU) using `python predict.py image.jpg --time`:

| Stage | Time |
|---|---|
| Image decode + resize to 512px | ~5 ms |
| FFT (numpy) | ~8 ms |
| All other features (OpenCV) | ~5 ms |
| Logistic regression score | < 1 ms |
| **Total end-to-end** | **~18–25 ms per image** |

This feels instant at human timescales. No GPU, no large model file, no
network call.

---

## Cost Per Image

| Deployment | Cost |
|---|---|
| **On-device (phone/laptop)** | **$0** — runs on CPU, no API, no server |
| Cloud CPU instance (e.g. AWS t3.medium, ~$0.04/hr) | ~$0.0003 per 1,000 images |
| Cloud CPU instance at scale (1M images/day) | ~$0.28 per million images |

Assumptions for cloud estimate: 25 ms/image → 40 images/second on one core →
~144,000 images/hour → one $0.04/hr instance handles ~3.6M images/day. At
that rate, cost is well under $0.001 per 1,000 images.

The on-device path (running directly on the user's phone) costs nothing and
adds zero latency overhead from a network round-trip — the ideal deployment
for a mobile app fraud check.

---

## What I Would Improve With More Time

1. **More data, more variety** — at least 500 images per class, spanning many
   screen types (AMOLED, LCD, e-ink, printed), angles, and lighting conditions.
2. **Add chromatic aberration feature** — re-photographing a screen through a
   lens produces lateral chromatic fringing at high-contrast edges that doesn't
   appear in direct photos. Strong, specific, hard to spoof.
3. **Add JPEG artifact density** — screen recaptures often go through two
   rounds of lossy compression, leaving a stronger DCT block-grid artefact
   pattern in the 8×8 frequency bands.
4. **Calibrate threshold on a held-out validation set** — the current threshold
   is picked on training data; a separate validation set would give a less
   optimistic operating point.
5. **Adversarial robustness** — if cheaters know the detector, they will add
   noise or use high-quality captures at short range. Periodic retraining on
   flagged borderline cases (active learning) keeps the detector current.
6. **Phone deployment** — convert features.py to C++ or use a CoreML/TFLite
   wrapper; the logistic regression is trivial to port. Total on-phone latency
   would be under 10 ms.
