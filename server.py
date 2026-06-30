"""
server.py — tiny Flask server to bridge the browser demo with predict.py

Usage:
    pip install flask pillow numpy opencv-python-headless
    python server.py        # serves on http://localhost:5000

Then open demo.html in your browser (or let the server serve it at /).
The demo page POSTs the image to /analyze and renders the JSON result live.
"""

import io
import time
import os
import sys
import tempfile

try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    print("ERROR: Flask not installed.  Run:  pip install flask")
    sys.exit(1)

from PIL import Image

# Resolve sibling modules regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import feature_vector
from model import load_model, score as model_score

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(here, "demo.html")
    return send_file(demo_path)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400

    file = request.files["image"]
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400

    t0 = time.perf_counter()
    vec, feats = feature_vector(img)
    m = get_model()
    s = model_score(vec, m)
    latency_ms = (time.perf_counter() - t0) * 1000

    threshold = m.get("threshold", 0.5)
    return jsonify({
        "score":       round(s, 4),
        "label":       "screen" if s >= threshold else "real",
        "threshold":   threshold,
        "latency_ms":  round(latency_ms, 2),
        "trained":     m.get("trained", False),
        "features":    {k: round(v, 5) for k, v in feats.items()},
    })


@app.route("/health")
def health():
    m = get_model()
    return jsonify({
        "status":   "ok",
        "trained":  m.get("trained", False),
        "cv_acc":   m.get("cv_accuracy", None),
        "n_real":   m.get("n_real", None),
        "n_screen": m.get("n_screen", None),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  🚀  Demo server at  http://localhost:{port}")
    print(f"  Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False)
