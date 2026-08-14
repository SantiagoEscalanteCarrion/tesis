"""
gradcam_evaluation.py — Evaluación cuantitativa de Grad-CAM sobre el test set
===============================================================================
Complementa la evaluación cualitativa de la Fig. 3 del paper con métricas
cuantitativas sobre las ~48 imágenes del test set fijo (seed=42):

  - IoU:           intersección / unión entre el mapa Grad-CAM binarizado
                   (top 20% de activación) y la región de interés anatómica.
  - Pointing game: si el píxel de máxima activación cae dentro de la ROI (0/1).

ROI anatómica: polígono convex-hull de los 4 keypoints del tronco
  (hombros izq/der + caderas izq/der) escalados a espacio de píxel (224×224).
  Se re-corre MediaPipe sobre las imágenes del test para obtener coordenadas
  raw (x,y) — split_features.pkl solo guarda las 12 features derivadas.

Implementación de Grad-CAM: idéntica a generate_paper_figures.py (model_a /
  model_b split, tape.watch, ReLU-weighted sum).

Outputs en outputs/paper_figures/gradcam_eval/:
  gradcam_metrics_test.csv   — métricas por imagen
  gradcam_summary.txt        — media±std por clase y global
  gradcam_examples.png       — 3 ejemplos visuales

Ejecutar desde la raíz del proyecto:
  python gradcam_evaluation.py
"""

import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DATASET_AUG_DIR, OUTPUT_DIR, IMG_SIZE, MP_LANDMARKS
from data_utils import grouped_split
from model_io import load_model_compat

OUT_DIR      = Path(OUTPUT_DIR) / "paper_figures" / "gradcam_eval"
TARGET_SIZE  = IMG_SIZE   # (224, 224)  — (W, H) para cv2.resize
GRADCAM_PCT  = 20         # top-20% de activación → máscara binaria


# ── Grad-CAM (patrón de generate_paper_figures.py) ────────────────────────

def _build_gradcam_models(cnn_model):
    """Divide E1 en model_a (backbone→top_activation) y model_b (head lineal).

    Replica exactamente plot_gradcam_panel de generate_paper_figures.py:
      model_a: backbone.input → top_activation  (7×7×1280)
      model_b: top_activation → Dense(1) sin sigmoid (evita saturación)
    """
    import tensorflow as tf

    backbone = cnn_model.get_layer("efficientnetb0")
    model_a  = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer("top_activation").output,
    )
    top_shape = model_a.output_shape[1:]  # (7, 7, 1280)
    _inp    = tf.keras.Input(shape=top_shape)
    _x      = cnn_model.get_layer("gap")(_inp)
    _x      = cnn_model.get_layer("batch_normalization")(_x)
    _x      = cnn_model.get_layer("dropout")(_x)
    _x      = cnn_model.get_layer("fc1")(_x)
    _x      = cnn_model.get_layer("dropout_1")(_x)
    _linear = tf.keras.layers.Dense(1, activation=None, name="output_linear")
    _x      = _linear(_x)
    model_b = tf.keras.Model(inputs=_inp, outputs=_x)
    _linear.set_weights(cnn_model.get_layer("output").get_weights())
    return model_a, model_b


def _compute_gradcam(model_a, model_b, cnn_model, img_path, label):
    """Calcula heatmap Grad-CAM para la clase verdadera de la imagen.

    Retorna (heatmap (224,224) float32 [0,1], img_rgb uint8 (224,224,3), prob float).
    Retorna (None, None, None) si la imagen no se puede leer.
    """
    import tensorflow as tf
    from matplotlib import cm

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None
    img_bgr_r = cv2.resize(img_bgr, TARGET_SIZE)
    img_rgb   = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB)

    # Preprocesar igual que en generate_paper_figures.py
    img_pre = tf.keras.applications.efficientnet.preprocess_input(
        img_rgb.astype(np.float32).copy()
    )
    img_arr = np.expand_dims(img_pre, 0)   # (1, 224, 224, 3)

    prob = float(cnn_model.predict(img_arr, verbose=0)[0, 0])

    # Grad-CAM respecto a la clase verdadera (igual que Fig. 3 del paper)
    scoliosis_class = (label == 1)
    with tf.GradientTape() as tape:
        conv_out = model_a(img_arr, training=False)   # (1, 7, 7, 1280)
        tape.watch(conv_out)
        logit    = model_b(conv_out, training=False)  # (1, 1)
        loss     = logit[:, 0] if scoliosis_class else -logit[:, 0]

    grads   = tape.gradient(loss, conv_out)                        # (1, 7, 7, 1280)
    alpha   = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)   # (1, 1, 1, 1280)
    heatmap = tf.nn.relu(tf.reduce_sum(alpha * conv_out, axis=-1)) # (1, 7, 7)
    heatmap = heatmap[0].numpy()                                   # (7, 7)
    heatmap = cv2.resize(heatmap, TARGET_SIZE)                     # (224, 224)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.astype(np.float32), img_rgb, prob


# ── ROI anatómica via MediaPipe ───────────────────────────────────────────

def _get_trunk_landmarks_px(img_path, landmarker):
    """Detecta pose y devuelve las coordenadas en píxeles de hombros+caderas.

    Retorna dict {"left_shoulder": (x,y), "right_shoulder": (x,y),
                  "left_hip": (x,y), "right_hip": (x,y)}
    o None si la detección falla.
    Coordenadas escaladas a espacio de píxel (224×224).
    """
    import mediapipe as mp

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    img_bgr_r = cv2.resize(img_bgr, TARGET_SIZE)
    img_rgb   = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    lms  = result.pose_landmarks[0]
    W, H = TARGET_SIZE          # (224, 224)
    idx  = MP_LANDMARKS

    try:
        pts = {
            "left_shoulder":  (int(lms[idx["left_shoulder"]].x  * W),
                               int(lms[idx["left_shoulder"]].y  * H)),
            "right_shoulder": (int(lms[idx["right_shoulder"]].x * W),
                               int(lms[idx["right_shoulder"]].y * H)),
            "left_hip":       (int(lms[idx["left_hip"]].x       * W),
                               int(lms[idx["left_hip"]].y       * H)),
            "right_hip":      (int(lms[idx["right_hip"]].x      * W),
                               int(lms[idx["right_hip"]].y      * H)),
        }
    except (IndexError, AttributeError):
        return None
    return pts


def _build_roi_mask(landmarks_px):
    """Crea máscara binaria (224,224) del convex-hull de hombros+caderas."""
    pts  = np.array(list(landmarks_px.values()), dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)   # (H, W)
    cv2.fillPoly(mask, [hull], 1)
    return mask


# ── Métricas ──────────────────────────────────────────────────────────────

def _compute_metrics(heatmap, roi_mask):
    """IoU (top-PCT% vs ROI) y pointing game."""
    thresh     = np.percentile(heatmap, 100 - GRADCAM_PCT)
    attn_mask  = (heatmap >= thresh).astype(np.uint8)
    inter      = int((attn_mask & roi_mask).sum())
    union      = int((attn_mask | roi_mask).sum())
    iou        = inter / (union + 1e-8)
    max_pt     = np.unravel_index(heatmap.argmax(), heatmap.shape)  # (row, col)
    pg_hit     = int(roi_mask[max_pt] == 1)
    return {"iou": float(iou), "pg_hit": pg_hit}


# ── Visualización ─────────────────────────────────────────────────────────

def _plot_examples(records, out_path, n=3):
    """Panel de n ejemplos: [original | Grad-CAM overlay | ROI superpuesta]."""
    from matplotlib import cm

    detected = [r for r in records if r["pose_detected"] and r["heatmap"] is not None]
    if not detected:
        print("[gradcam] No hay ejemplos con pose detectada para visualizar.")
        return

    # 1 scoliosis, 1 sano, 1 scoliosis extra
    sco  = [r for r in detected if r["label"] == 1]
    heal = [r for r in detected if r["label"] == 0]
    examples = []
    if sco:            examples.append(sco[0])
    if heal:           examples.append(heal[0])
    if len(sco) > 1:   examples.append(sco[1])
    examples = examples[:n]

    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 4 * len(examples)))
    fig.patch.set_facecolor("#f8f9fa")
    if len(examples) == 1:
        axes = [axes]

    for row_axes, rec in zip(axes, examples):
        img_rgb  = rec["img_rgb"]
        heatmap  = rec["heatmap"]
        roi_mask = rec["roi_mask"]
        label_str = "Scoliosis" if rec["label"] == 1 else "Healthy"

        # Panel 1: imagen original
        row_axes[0].imshow(img_rgb)
        row_axes[0].set_title(f"Original ({label_str})\nProb={rec['prob']:.2f}",
                               fontsize=10)
        row_axes[0].axis("off")

        # Panel 2: Grad-CAM overlay
        colored  = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay  = cv2.addWeighted(img_bgr, 0.55, colored, 0.45, 0)
        overlay  = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        row_axes[1].imshow(overlay)
        pg_sym = "✓" if rec["pg_hit"] else "✗"
        row_axes[1].set_title(
            f"Grad-CAM\nIoU={rec['iou']:.3f}  PG={pg_sym}", fontsize=10
        )
        row_axes[1].axis("off")

        # Panel 3: ROI anatómica superpuesta
        roi_vis  = img_rgb.copy()
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(roi_vis, contours, -1, (255, 165, 0), 2)
        row_axes[2].imshow(roi_vis)
        row_axes[2].set_title("Anatomical ROI\n(shoulders + hips)", fontsize=10)
        row_axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[gradcam] Ejemplos → {out_path}")


# ── Core ──────────────────────────────────────────────────────────────────

def run_gradcam_evaluation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Modelos ───────────────────────────────────────────────────────────
    cnn_path = Path(OUTPUT_DIR) / "cnn" / "cnn_final.keras"
    print(f"[gradcam] Cargando {cnn_path} ...")
    cnn_model = load_model_compat(str(cnn_path))
    model_a, model_b = _build_gradcam_models(cnn_model)

    # ── Landmarker ────────────────────────────────────────────────────────
    from model_pose import _get_landmarker
    print("[gradcam] Inicializando MediaPipe landmarker ...")
    landmarker = _get_landmarker()

    # ── Test set ──────────────────────────────────────────────────────────
    test_split  = grouped_split(DATASET_AUG_DIR)["test"]
    test_paths  = [p for p, _ in test_split]
    test_labels = [l for _, l in test_split]
    print(f"[gradcam] Test set: {len(test_paths)} imágenes.")

    # ── Loop principal ────────────────────────────────────────────────────
    records = []
    for i, (path, label) in enumerate(zip(test_paths, test_labels)):
        fname = Path(path).name
        print(f"  [{i+1:02d}/{len(test_paths)}] {fname} ...", end=" ")

        heatmap, img_rgb, prob = _compute_gradcam(model_a, model_b, cnn_model,
                                                   path, label)
        if heatmap is None:
            print("ERROR leyendo imagen")
            continue

        lm_px    = _get_trunk_landmarks_px(path, landmarker)
        pose_det = lm_px is not None
        iou = pg_hit = float("nan")
        roi_mask = None

        if pose_det:
            roi_mask = _build_roi_mask(lm_px)
            metrics  = _compute_metrics(heatmap, roi_mask)
            iou      = metrics["iou"]
            pg_hit   = metrics["pg_hit"]
            print(f"IoU={iou:.3f}  PG={int(pg_hit)}")
        else:
            pg_hit = 0
            print("pose no detectada")

        records.append({
            "path":          path,
            "label":         label,
            "prob":          prob,
            "iou":           iou,
            "pg_hit":        int(pg_hit),
            "pose_detected": pose_det,
            "heatmap":       heatmap,
            "img_rgb":       img_rgb,
            "roi_mask":      roi_mask,
        })

    landmarker.close()

    _save_csv(records)
    _save_summary(records)
    _plot_examples(records, OUT_DIR / "gradcam_examples.png")
    return records


# ── Guardar resultados ─────────────────────────────────────────────────────

def _save_csv(records):
    csv_path = OUT_DIR / "gradcam_metrics_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "prob", "iou", "pg_hit", "pose_detected"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "path":          r["path"],
                "label":         r["label"],
                "prob":          f"{r['prob']:.4f}",
                "iou":           f"{r['iou']:.4f}" if not np.isnan(r["iou"]) else "nan",
                "pg_hit":        r["pg_hit"],
                "pose_detected": int(r["pose_detected"]),
            })
    print(f"[gradcam] CSV → {csv_path}")


def _save_summary(records):
    detected = [r for r in records if r["pose_detected"]]
    lines    = []
    lines.append("=" * 62)
    lines.append("Quantitative Grad-CAM Evaluation — E1 EfficientNetB0 CNN")
    lines.append("=" * 62)
    lines.append(f"  Test set images:      {len(records)}")
    lines.append(f"  Pose detected:        {len(detected)}")
    lines.append(f"  Pose failed (skipped): {len(records) - len(detected)}")
    lines.append(f"  Threshold:            top-{GRADCAM_PCT}% activation")
    lines.append(f"  ROI:                  convex hull (shoulders + hips)")
    lines.append("")

    for cls_label, cls_filter in [
        ("Global",            None),
        ("Scoliosis (1)",     1),
        ("Healthy (0)",       0),
    ]:
        subset = detected if cls_filter is None else [
            r for r in detected if r["label"] == cls_filter
        ]
        if not subset:
            continue
        ious  = [r["iou"] for r in subset if not np.isnan(r["iou"])]
        pg    = [r["pg_hit"] for r in subset]
        lines.append(f"  {cls_label} (n={len(subset)})")
        if ious:
            lines.append(f"    IoU (top-{GRADCAM_PCT}%): "
                         f"{np.mean(ious):.3f} ± {np.std(ious):.3f}  "
                         f"[min={min(ious):.3f}, max={max(ious):.3f}]")
        lines.append(f"    Pointing game:  {100*np.mean(pg):.1f}%  "
                     f"({sum(pg)}/{len(pg)} hits)")
        lines.append("")

    txt = "\n".join(lines)
    print("\n" + txt)
    txt_path = OUT_DIR / "gradcam_summary.txt"
    txt_path.write_text(txt, encoding="utf-8")
    print(f"[gradcam] Resumen → {txt_path}")


if __name__ == "__main__":
    run_gradcam_evaluation()
