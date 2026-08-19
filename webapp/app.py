"""
webapp/app.py — FastAPI backend para demo de detección de escoliosis.

Endpoints:
  GET  /           → página HTML
  POST /predict    → JSON con resultados E1/E2/E3 + Grad-CAM en base64

Arrancar desde la raíz del proyecto (tesis/):
  uvicorn webapp.app:app --reload --host 0.0.0.0 --port 7860
"""

import os
import sys
import base64
import pickle
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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
    from model_io import load_model_compat
    from model_pose import _get_landmarker

    print("[webapp] Cargando modelos...")

    _m["e1"] = load_model_compat(os.path.join(OUTPUT_DIR, "cnn", "cnn_final.keras"))
    with open(os.path.join(OUTPUT_DIR, "pose", "pose_xgboost.pkl"), "rb") as f:
        _m["e2"] = pickle.load(f)

    _m["e3"] = load_model_compat(os.path.join(OUTPUT_DIR, "hybrid", "hybrid_final.keras"))
    with open(os.path.join(OUTPUT_DIR, "hybrid", "pose_norm_stats.pkl"), "rb") as f:
        _m["e3_norm"] = pickle.load(f)

    temp_path = os.path.join(OUTPUT_DIR, "hybrid", "e3_temperature.pkl")
    if os.path.exists(temp_path):
        with open(temp_path, "rb") as f:
            T_loaded = pickle.load(f)["T"]
        if T_loaded >= 1.0:
            _m["e3_T"] = T_loaded
            print(f"[webapp] Temperature scaling E3: T={_m['e3_T']:.4f}")
        else:
            _m["e3_T"] = 1.0
            print(f"[webapp] T cargado={T_loaded:.4f} < 1, usando T=1.0")
    else:
        _m["e3_T"] = 1.0
        print("[webapp] Sin calibración de temperatura (T=1.0)")

    _m["lm"] = _get_landmarker()

    # ── Cachear sub-modelos para Grad-CAM (evitar reconstruir en cada request) ──
    e1       = _m["e1"]
    backbone = e1.get_layer("efficientnetb0")
    model_a  = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer("top_activation").output,
    )
    top_shape = model_a.output_shape[1:]
    _inp = tf.keras.Input(shape=top_shape)
    _x   = e1.get_layer("gap")(_inp)
    _x   = e1.get_layer("batch_normalization")(_x)
    _x   = e1.get_layer("dropout")(_x)
    _x   = e1.get_layer("fc1")(_x)
    _x   = e1.get_layer("dropout_1")(_x)
    _lin = tf.keras.layers.Dense(1, activation=None, name="out_lin")
    _x   = _lin(_x)
    model_b = tf.keras.Model(inputs=_inp, outputs=_x)
    _lin.set_weights(e1.get_layer("output").get_weights())
    _m["e1_model_a"] = model_a
    _m["e1_model_b"] = model_b

    print("[webapp] Modelos listos.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    if "lm" in _m:
        _m["lm"].close()


app = FastAPI(title="Detección de Escoliosis con IA", lifespan=lifespan)
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


def _apply_noise(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """Agrega ruido gaussiano a la imagen (mismo método que en noise_robustness.py)."""
    if sigma <= 0.0:
        return img_bgr
    noise = np.random.normal(0.0, sigma * 255.0, img_bgr.shape).astype(np.float32)
    return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _pose_features(img_bgr: np.ndarray):
    """Extrae los 12 features de pose o devuelve None si no detecta pose."""
    from model_pose import extract_pose_features_from_image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        path = tmp.name
    try:
        return extract_pose_features_from_image(path, _m["lm"])
    finally:
        os.unlink(path)


def _pose_landmarks_px(img_bgr: np.ndarray, size: int = 224):
    """Retorna [(x_px, y_px)] para landmarks de hombros+caderas (11,12,23,24) o None."""
    import mediapipe as mp
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result  = _m["lm"].detect(mp_img)
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    return [(int(lms[i].x * size), int(lms[i].y * size)) for i in [11, 12, 23, 24]]


def _gradcam_b64(img_bgr: np.ndarray, roi_pts=None) -> str:
    """Grad-CAM sobre E1 con ROI anatómico opcional. Devuelve base64 PNG."""
    import tensorflow as tf
    from matplotlib import cm

    img_arr = _preprocess(img_bgr)

    with tf.GradientTape() as tape:
        conv  = _m["e1_model_a"](img_arr, training=False)
        tape.watch(conv)
        logit = _m["e1_model_b"](conv, training=False)
        loss  = logit[:, 0]

    grads   = tape.gradient(loss, conv)
    alpha   = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(alpha * conv, axis=-1))[0].numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    colored     = (cm.jet(cv2.resize(heatmap, IMG_SIZE))[:, :, :3] * 255).astype(np.uint8)
    overlay     = cv2.addWeighted(img_rgb, 0.55, colored, 0.45, 0)

    # Dibujar contorno del ROI anatómico (naranja) si se detectó pose
    if roi_pts is not None:
        hull = cv2.convexHull(np.array(roi_pts, dtype=np.int32))
        cv2.polylines(overlay, [hull], True, (255, 165, 0), 3)

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    noise_sigma: float = Form(0.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se aceptan imágenes (JPEG, PNG).")

    raw     = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "No se pudo decodificar la imagen.")

    # Aplicar ruido gaussiano (para la demo de robustez con el slider)
    sigma = max(0.0, min(float(noise_sigma), 1.0))
    if sigma > 0.0:
        img_bgr = _apply_noise(img_bgr, sigma)

    result = {}

    # ── E1 CNN ─────────────────────────────────────────────────
    img_arr = _preprocess(img_bgr)
    prob_e1 = float(_m["e1"].predict(img_arr, verbose=0)[0, 0])
    roi_pts = _pose_landmarks_px(img_bgr)
    result["e1"] = {
        "probability": round(prob_e1, 4),
        "prediction":  "Scoliosis" if prob_e1 >= 0.5 else "Healthy",
        "gradcam_b64": _gradcam_b64(img_bgr, roi_pts),
    }

    # ── E2 XGBoost ─────────────────────────────────────────────
    feats = _pose_features(img_bgr)
    if feats is not None:
        prob_e2 = float(_m["e2"].predict_proba([feats])[0, 1])
        result["e2"] = {
            "probability":   round(prob_e2, 4),
            "prediction":    "Scoliosis" if prob_e2 >= 0.5 else "Healthy",
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
        eps     = 1e-7
        logit   = float(np.log(np.clip(raw_prob, eps, 1 - eps) /
                               (1 - np.clip(raw_prob, eps, 1 - eps))))
        prob_e3 = float(1 / (1 + np.exp(-logit / _m["e3_T"])))
        result["e3"] = {
            "probability": round(prob_e3, 4),
            "prediction":  "Scoliosis" if prob_e3 >= 0.5 else "Healthy",
        }
    else:
        result["e3"] = {"probability": None, "prediction": "N/A"}

    # ── Consenso por mayoría ────────────────────────────────────
    votes = [v["prediction"] for v in result.values() if v.get("prediction") != "N/A"]
    result["consensus"] = "Scoliosis" if votes.count("Scoliosis") >= 2 else "Healthy"

    return result
