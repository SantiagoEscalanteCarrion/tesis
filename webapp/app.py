"""
webapp/app.py — FastAPI backend para demo de detección de escoliosis.

Endpoints:
  GET  /           → página HTML
  POST /predict    → JSON con resultados E1/E2/E3 + Grad-CAM en base64

Arrancar desde la raíz del proyecto (tesis/):
  uvicorn webapp.app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import base64
import pickle
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Rutas ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # tesis/
sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR, IMG_SIZE, NUM_POSE_FEATURES

# ── Estado global ──────────────────────────────────────────────
_m: dict = {}


def _load_models() -> None:
    import tensorflow as tf
    from model_pose import _get_landmarker

    print("[webapp] Cargando modelos...")

    _m["e1"] = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "cnn", "cnn_final.keras")
    )
    with open(os.path.join(OUTPUT_DIR, "pose", "pose_xgboost.pkl"), "rb") as f:
        _m["e2"] = pickle.load(f)

    _m["e3"] = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "hybrid", "hybrid_final.keras")
    )
    with open(os.path.join(OUTPUT_DIR, "hybrid", "pose_norm_stats.pkl"), "rb") as f:
        _m["e3_norm"] = pickle.load(f)

    temp_path = os.path.join(OUTPUT_DIR, "hybrid", "e3_temperature.pkl")
    if os.path.exists(temp_path):
        with open(temp_path, "rb") as f:
            _m["e3_T"] = pickle.load(f)["T"]
        print(f"[webapp] Temperature scaling E3: T={_m['e3_T']:.4f}")
    else:
        _m["e3_T"] = 1.0
        print("[webapp] Temperature scaling E3: no calibrado (T=1.0)")

    _m["lm"] = _get_landmarker()
    print("[webapp] Modelos listos.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    if "lm" in _m:
        _m["lm"].close()


app = FastAPI(title="Scoliosis Detection Demo", lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


# ── Utilidades ─────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> "np.ndarray":
    """BGR uint8 → tensor EfficientNet (1, H, W, 3) float32."""
    import tensorflow as tf
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, 0)


def _pose_features(img_bgr: np.ndarray):
    """Extrae las 12 features de pose o devuelve None si no detecta pose."""
    from model_pose import extract_pose_features_from_image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        path = tmp.name
    try:
        return extract_pose_features_from_image(path, _m["lm"])
    finally:
        os.unlink(path)


def _gradcam_b64(img_bgr: np.ndarray) -> str:
    """Grad-CAM sobre E1. Devuelve base64 PNG del overlay."""
    import tensorflow as tf
    from matplotlib import cm

    img_arr  = _preprocess(img_bgr)
    model    = _m["e1"]
    backbone = model.get_layer("efficientnetb0")

    model_a = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer("top_activation").output,
    )
    top_shape = model_a.output_shape[1:]
    _inp = tf.keras.Input(shape=top_shape)
    _x   = model.get_layer("gap")(_inp)
    _x   = model.get_layer("batch_normalization")(_x)
    _x   = model.get_layer("dropout")(_x)
    _x   = model.get_layer("fc1")(_x)
    _x   = model.get_layer("dropout_1")(_x)
    _lin = tf.keras.layers.Dense(1, activation=None, name="out_lin")
    _x   = _lin(_x)
    model_b = tf.keras.Model(inputs=_inp, outputs=_x)
    _lin.set_weights(model.get_layer("output").get_weights())

    with tf.GradientTape() as tape:
        conv  = model_a(img_arr, training=False)
        tape.watch(conv)
        logit = model_b(conv, training=False)
        loss  = logit[:, 0]

    grads   = tape.gradient(loss, conv)
    alpha   = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(alpha * conv, axis=-1))[0].numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    colored     = (cm.jet(cv2.resize(heatmap, IMG_SIZE))[:, :, :3] * 255).astype(np.uint8)
    overlay     = cv2.addWeighted(img_rgb, 0.55, colored, 0.45, 0)

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se aceptan imágenes (JPEG, PNG).")

    raw     = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "No se pudo decodificar la imagen.")

    result = {}

    # ── E1 CNN ─────────────────────────────────────────────────
    img_arr = _preprocess(img_bgr)
    prob_e1 = float(_m["e1"].predict(img_arr, verbose=0)[0, 0])
    result["e1"] = {
        "probability": round(prob_e1, 4),
        "prediction":  "Scoliosis" if prob_e1 >= 0.5 else "Healthy",
        "gradcam_b64": _gradcam_b64(img_bgr),
    }

    # ── E2 XGBoost ─────────────────────────────────────────────
    feats = _pose_features(img_bgr)
    if feats is not None:
        prob_e2 = float(_m["e2"].predict_proba([feats])[0, 1])
        result["e2"] = {
            "probability":  round(prob_e2, 4),
            "prediction":   "Scoliosis" if prob_e2 >= 0.5 else "Healthy",
            "pose_detected": True,
        }
    else:
        result["e2"] = {"probability": None, "prediction": "N/A", "pose_detected": False}

    # ── E3 Hybrid ──────────────────────────────────────────────
    if feats is not None:
        norm       = _m["e3_norm"]
        feats_norm = ((feats - norm["mean"]) / norm["std"]).astype(np.float32)
        raw_prob   = float(
            _m["e3"].predict(
                [img_arr, np.expand_dims(feats_norm, 0)], verbose=0
            )[0, 0]
        )
        # Temperature scaling: sigmoid(logit / T)
        eps      = 1e-7
        logit    = float(np.log(np.clip(raw_prob, eps, 1 - eps) /
                                (1 - np.clip(raw_prob, eps, 1 - eps))))
        prob_e3  = float(1 / (1 + np.exp(-logit / _m["e3_T"])))
        result["e3"] = {
            "probability": round(prob_e3, 4),
            "prediction":  "Scoliosis" if prob_e3 >= 0.5 else "Healthy",
        }
    else:
        result["e3"] = {"probability": None, "prediction": "N/A"}

    # ── Consenso por mayoría ────────────────────────────────────
    votes = [v["prediction"] for v in result.values() if v["prediction"] != "N/A"]
    result["consensus"] = "Scoliosis" if votes.count("Scoliosis") >= 2 else "Healthy"

    return result
