"""
visualize.py — Análisis Clínico Visual por Modelo
===================================================
Genera una imagen de diagnóstico mostrando CÓMO cada modelo
tomó su decisión sobre una imagen de entrada.

  E1 (CNN)    → Grad-CAM: zonas de la imagen que activaron la predicción
  E2 (Pose)   → Landmarks corporales + líneas de asimetría + valores de features
  E3 (Hybrid) → Combinación de Grad-CAM + landmarks de pose

Uso:
    python visualize.py --model cnn   --image ruta/imagen.jpg
    python visualize.py --model pose  --image ruta/imagen.jpg
    python visualize.py --model hybrid --image ruta/imagen.jpg

    # Con rutas explícitas a los modelos:
    python visualize.py --model cnn --image img.jpg \
        --model-path outputs/cnn/cnn_final.keras
"""

import argparse
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import tensorflow as tf

from config import IMG_SIZE, OUTPUT_DIR, CLASS_LABELS, POSE_FEATURE_NAMES, MP_LANDMARKS


# ─────────────────────────────────────────────────────────────
# UTILIDADES COMUNES
# ─────────────────────────────────────────────────────────────

def _load_image_cv2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return img


def _lm_px(lm, w, h):
    """Convierte landmark normalizado (0-1) a coordenadas de píxel."""
    return (int(lm.x * w), int(lm.y * h))


def _draw_dashed_line(img, pt1, pt2, color, thickness=2, gap=10):
    """Dibuja una línea discontinua entre dos puntos."""
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    pts = int(dist / gap)
    if pts == 0:
        return
    for i in range(pts):
        s = i / pts
        e = min((i + 0.5) / pts, 1.0)
        p1 = (int(pt1[0] + s * (pt2[0] - pt1[0])),
              int(pt1[1] + s * (pt2[1] - pt1[1])))
        p2 = (int(pt1[0] + e * (pt2[0] - pt1[0])),
              int(pt1[1] + e * (pt2[1] - pt1[1])))
        cv2.line(img, p1, p2, color, thickness)


def _predict_cnn(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    prob = float(model.predict(arr, verbose=0)[0, 0])
    return prob, arr


# ─────────────────────────────────────────────────────────────
# E1: VISUALIZACIÓN CNN — GRAD-CAM CLÍNICO
# ─────────────────────────────────────────────────────────────

def visualize_cnn(model, image_path, output_dir=None,
                  layer_name="top_activation", threshold=0.5):
    """
    Genera una figura de 3 paneles:
      1. Imagen original con anotación de predicción
      2. Mapa de calor Grad-CAM puro
      3. Superposición: imagen + Grad-CAM

    El mapa de calor muestra qué zonas de la espalda
    el modelo usó para tomar su decisión.
    """
    # Grad-CAM
    prob, arr = _predict_cnn(model, image_path)
    label = CLASS_LABELS[int(prob >= threshold)]

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    # Imagen original
    orig = cv2.imread(image_path)
    orig_rgb = cv2.cvtColor(cv2.resize(orig, IMG_SIZE), cv2.COLOR_BGR2RGB)

    # Heatmap coloreado
    heat_resized = cv2.resize(heatmap, IMG_SIZE)
    heat_colored = cm.jet(heat_resized)[:, :, :3]

    # Superposición
    overlay = (0.55 * orig_rgb / 255.0 + 0.45 * heat_colored)
    overlay = np.clip(overlay, 0, 1)

    # ── Figura ────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor("#1a1a2e")

    color = "#e63946" if label == "scoliosis_yes" else "#52b788"
    title_label = "ESCOLIOSIS DETECTADA" if label == "scoliosis_yes" else "SIN ESCOLIOSIS"

    fig.suptitle(
        f"Análisis CNN (Grad-CAM)  —  {title_label}  (confianza: {prob:.1%})",
        fontsize=14, fontweight="bold", color=color, y=1.01
    )

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(orig_rgb)
    ax1.set_title("Imagen original", color="white", fontsize=11)
    ax1.axis("off")
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02", linewidth=3,
        edgecolor=color, facecolor="none",
        transform=ax1.transAxes
    )
    ax1.add_patch(rect)

    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(heat_resized, cmap="jet", vmin=0, vmax=1)
    ax2.set_title("Mapa de activación (Grad-CAM)", color="white", fontsize=11)
    ax2.axis("off")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Importancia", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Superposición — zona de decisión", color="white", fontsize=11)
    ax3.axis("off")

    # Leyenda de interpretación
    fig.text(0.5, -0.04,
             "Rojo/Amarillo = zonas de alta activación que influyeron en la predicción\n"
             "Azul/Verde = zonas ignoradas por el modelo",
             ha="center", fontsize=9, color="#aaaaaa",
             style="italic")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(image_path).replace(".jpg", "_cnn_analysis.png")
        fig.savefig(os.path.join(output_dir, fname), dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Guardado: {os.path.join(output_dir, fname)}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# E2: VISUALIZACIÓN POSE — LANDMARKS + LÍNEAS CLÍNICAS
# ─────────────────────────────────────────────────────────────

def _extract_features_and_draw(image_path, landmarker, pipeline=None,
                                norm_stats=None, threshold=0.5):
    """
    Extrae pose features, dibuja landmarks y líneas clínicas sobre la imagen.
    Retorna (img_annotated_rgb, features_dict, label, prob).
    """
    from model_pose import extract_pose_features_from_image
    import mediapipe as mp

    img_bgr = _load_image_cv2(image_path)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    canvas = img_bgr.copy()
    features_dict = {}
    label, prob = "no_pose", 0.0

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        lm = result.pose_landmarks[0]
        idx = MP_LANDMARKS

        # Coordenadas en píxeles
        ls = _lm_px(lm[idx["left_shoulder"]],  w, h)
        rs = _lm_px(lm[idx["right_shoulder"]], w, h)
        le = _lm_px(lm[idx["left_elbow"]],     w, h)
        re = _lm_px(lm[idx["right_elbow"]],    w, h)
        lw = _lm_px(lm[idx["left_wrist"]],     w, h)
        rw = _lm_px(lm[idx["right_wrist"]],    w, h)
        lh = _lm_px(lm[idx["left_hip"]],       w, h)
        rh = _lm_px(lm[idx["right_hip"]],      w, h)

        mid_s = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        mid_h = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

        # ── Líneas clínicas ───────────────────────────────────

        # Línea horizontal de hombros (blanca)
        cv2.line(canvas, ls, rs, (255, 255, 255), 2)
        # Línea horizontal de caderas (blanca)
        cv2.line(canvas, lh, rh, (255, 255, 255), 2)
        # Línea vertical de referencia: midHombro → midCadera (naranja)
        cv2.line(canvas, mid_s, mid_h, (0, 140, 255), 3)
        # Líneas de codos y muñecas (gris discontinuo)
        _draw_dashed_line(canvas, le, re, (180, 180, 180), 1, 8)
        _draw_dashed_line(canvas, lw, rw, (180, 180, 180), 1, 8)

        # ── Flechas de asimetría ──────────────────────────────
        def _asym_arrow(ptA, ptB, side_x):
            """Dibuja flecha doble indicando diferencia de altura."""
            top = min(ptA[1], ptB[1])
            bot = max(ptA[1], ptB[1])
            if bot - top < 4:
                return
            x = side_x
            cv2.arrowedLine(canvas, (x, bot), (x, top),
                            (0, 255, 255), 2, tipLength=0.3)
            cv2.arrowedLine(canvas, (x, top), (x, bot),
                            (0, 255, 255), 2, tipLength=0.3)

        _asym_arrow(ls, rs, max(ls[0], rs[0]) + 18)
        _asym_arrow(lh, rh, max(lh[0], rh[0]) + 18)

        # ── Landmarks (círculos) ──────────────────────────────
        SHOULDER_COLOR = (0, 0, 220)   # rojo
        HIP_COLOR      = (220, 0, 0)   # azul
        ELBOW_COLOR    = (0, 180, 255) # amarillo-naranja
        WRIST_COLOR    = (180, 255, 0) # verde-amarillo

        for pt, col in [(ls, SHOULDER_COLOR), (rs, SHOULDER_COLOR),
                        (lh, HIP_COLOR),      (rh, HIP_COLOR),
                        (le, ELBOW_COLOR),    (re, ELBOW_COLOR),
                        (lw, WRIST_COLOR),    (rw, WRIST_COLOR)]:
            cv2.circle(canvas, pt, 9,  col,   -1)
            cv2.circle(canvas, pt, 9,  (255, 255, 255), 1)

        # ── Features numéricas ────────────────────────────────
        feats = extract_pose_features_from_image(image_path, landmarker)
        if feats is not None:
            features_dict = dict(zip(POSE_FEATURE_NAMES, feats))

        # ── Predicción ────────────────────────────────────────
        if pipeline is not None and feats is not None:
            f = feats.reshape(1, -1)
            if norm_stats:
                f = (f - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
            prob = float(pipeline.predict_proba(f)[0, 1])
            label = CLASS_LABELS[int(prob >= threshold)]

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas_rgb, features_dict, label, prob


def visualize_pose(landmarker, image_path, pipeline=None,
                   norm_stats=None, output_dir=None, threshold=0.5):
    """
    Genera figura de análisis de pose con:
      - Imagen con landmarks y líneas clínicas (izquierda)
      - Panel de barras con los 12 valores de features (derecha)
      - Predicción del clasificador ML
    """
    canvas_rgb, feats, label, prob = _extract_features_and_draw(
        image_path, landmarker, pipeline, norm_stats, threshold
    )

    color = "#e63946" if label == "scoliosis_yes" else "#52b788"
    title_label = "ESCOLIOSIS DETECTADA" if label == "scoliosis_yes" else "SIN ESCOLIOSIS"

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#1a1a2e")

    if label not in ("no_pose",):
        fig.suptitle(
            f"Análisis Pose (MediaPipe + ML)  —  {title_label}  (confianza: {prob:.1%})",
            fontsize=14, fontweight="bold", color=color, y=1.01
        )
    else:
        fig.suptitle("Análisis Pose — No se detectó pose en la imagen",
                     fontsize=14, color="#aaaaaa")

    # Panel izquierdo: imagen anotada
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(canvas_rgb)
    ax1.set_title("Landmarks y líneas de referencia", color="white", fontsize=11)
    ax1.axis("off")
    ax1.set_facecolor("#1a1a2e")
    rect = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01", linewidth=3,
        edgecolor=color, facecolor="none",
        transform=ax1.transAxes
    )
    ax1.add_patch(rect)

    # Leyenda de colores de landmarks
    legend_items = [
        mpatches.Patch(color="#dc143c", label="Hombros"),
        mpatches.Patch(color="#1e90ff", label="Caderas"),
        mpatches.Patch(color="#ffa500", label="Codos"),
        mpatches.Patch(color="#adff2f", label="Muñecas"),
        mpatches.Patch(color="white",   label="Línea hombros/caderas"),
        mpatches.Patch(color="orange",  label="Eje de columna"),
        mpatches.Patch(color="cyan",    label="Indicadores de asimetría"),
    ]
    ax1.legend(handles=legend_items, loc="lower left",
               fontsize=7.5, framealpha=0.6,
               facecolor="#222244", labelcolor="white")

    # Panel derecho: barras de features
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#1a1a2e")

    if feats:
        names  = list(feats.keys())
        values = list(feats.values())

        # Colorear según magnitud (mayor = más sospechoso)
        max_v = max(abs(v) for v in values) + 1e-8
        bar_colors = [cm.RdYlGn_r(abs(v) / max_v) for v in values]

        bars = ax2.barh(names, values, color=bar_colors, edgecolor="#333355",
                        height=0.65)
        for bar, v in zip(bars, values):
            ax2.text(max(v, 0) + max_v * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=8, color="white")

        ax2.set_xlim(0, max_v * 1.25)
        ax2.set_xlabel("Valor de feature", color="white", fontsize=10)
        ax2.set_title("Features clínicas extraídas\n(rojo = mayor asimetría)",
                      color="white", fontsize=11)
        ax2.tick_params(colors="white", labelsize=8)
        ax2.spines[:].set_color("#444466")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")

        # Línea vertical en 0 de referencia
        ax2.axvline(0, color="white", linewidth=0.5, alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "No se extrajeron features\n(pose no detectada)",
                 ha="center", va="center", color="#aaaaaa", fontsize=12,
                 transform=ax2.transAxes)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(image_path).replace(".jpg", "_pose_analysis.png")
        fig.savefig(os.path.join(output_dir, fname), dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Guardado: {os.path.join(output_dir, fname)}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# E3: VISUALIZACIÓN HÍBRIDA — GRAD-CAM + LANDMARKS
# ─────────────────────────────────────────────────────────────

def visualize_hybrid(cnn_model, landmarker, image_path,
                     norm_stats=None, output_dir=None,
                     layer_name="top_activation", threshold=0.5):
    """
    Genera figura de 3 paneles:
      1. Landmarks + líneas clínicas (stream de Pose)
      2. Grad-CAM del stream CNN
      3. Fusión: landmarks superpuestos al Grad-CAM

    Muestra visualmente los dos streams que el modelo híbrido combina.
    """
    # Stream Pose
    canvas_rgb, feats, _, _ = _extract_features_and_draw(
        image_path, landmarker, norm_stats=norm_stats
    )

    # Stream CNN — Grad-CAM
    prob_cnn, arr = _predict_cnn(cnn_model, image_path)

    grad_model = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=[cnn_model.get_layer(layer_name).output, cnn_model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    heat_resized = cv2.resize(heatmap, (canvas_rgb.shape[1], canvas_rgb.shape[0]))

    # Fusión: canvas de pose + heatmap semitransparente
    heat_colored = cm.jet(heat_resized)[:, :, :3]
    canvas_f = canvas_rgb.astype(np.float32) / 255.0
    fusion = np.clip(0.65 * canvas_f + 0.35 * heat_colored, 0, 1)

    label = CLASS_LABELS[int(prob_cnn >= threshold)]
    color = "#e63946" if label == "scoliosis_yes" else "#52b788"
    title_label = "ESCOLIOSIS DETECTADA" if label == "scoliosis_yes" else "SIN ESCOLIOSIS"

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Análisis Híbrido (CNN + Pose)  —  {title_label}  (CNN confianza: {prob_cnn:.1%})",
        fontsize=14, fontweight="bold", color=color, y=1.01
    )

    titles = [
        "Stream B: Pose — Landmarks clínicos",
        "Stream A: CNN — Grad-CAM",
        "Fusión de ambos streams",
    ]
    images = [canvas_rgb, cm.jet(heat_resized)[:, :, :3], fusion]

    for i, (ax_img, title) in enumerate(zip(images, titles), 1):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(ax_img)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    # Anotación de features clave en el tercer panel
    ax3 = fig.axes[2]
    if feats:
        top_features = sorted(feats.items(), key=lambda kv: abs(kv[1]),
                               reverse=True)[:4]
        feat_text = "\n".join([f"{k}: {v:.4f}" for k, v in top_features])
        ax3.text(0.02, 0.02, f"Top 4 features (Pose):\n{feat_text}",
                 transform=ax3.transAxes, fontsize=7.5,
                 color="white", verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#222244", alpha=0.8, edgecolor="#555577"))

    fig.text(0.5, -0.03,
             "El modelo híbrido combina los dos streams para su predicción final",
             ha="center", fontsize=9, color="#aaaaaa", style="italic")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(image_path).replace(".jpg", "_hybrid_analysis.png")
        fig.savefig(os.path.join(output_dir, fname), dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Guardado: {os.path.join(output_dir, fname)}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _default_model_path(model_type):
    paths = {
        "cnn":    os.path.join(OUTPUT_DIR, "cnn",    "cnn_final.keras"),
        "pose":   os.path.join(OUTPUT_DIR, "pose",   "pose_RandomForest.pkl"),
        "hybrid": os.path.join(OUTPUT_DIR, "hybrid", "hybrid_final.keras"),
    }
    return paths.get(model_type, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualización clínica del análisis de escoliosis"
    )
    parser.add_argument("--model", choices=["cnn", "pose", "hybrid"], required=True)
    parser.add_argument("--image", required=True, help="Ruta a la imagen JPG")
    parser.add_argument("--model-path", default=None,
                        help="Ruta al archivo del modelo (opcional)")
    parser.add_argument("--norm-stats", default=None,
                        help="Ruta a pose_norm_stats.pkl (para pose/hybrid)")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "visualizations"),
                        help="Carpeta de salida para las imágenes")
    args = parser.parse_args()

    model_path = args.model_path or _default_model_path(args.model)

    norm_stats = None
    if args.norm_stats:
        with open(args.norm_stats, "rb") as f:
            norm_stats = pickle.load(f)
    elif args.model in ("pose", "hybrid"):
        default_ns = os.path.join(OUTPUT_DIR, "hybrid", "pose_norm_stats.pkl")
        if os.path.exists(default_ns):
            with open(default_ns, "rb") as f:
                norm_stats = pickle.load(f)

    if args.model == "cnn":
        model = tf.keras.models.load_model(model_path)
        visualize_cnn(model, args.image, output_dir=args.output)

    elif args.model == "pose":
        from model_pose import _get_landmarker
        landmarker = _get_landmarker()

        pipeline = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)

        visualize_pose(landmarker, args.image,
                       pipeline=pipeline, norm_stats=norm_stats,
                       output_dir=args.output)
        landmarker.close()

    elif args.model == "hybrid":
        from model_pose import _get_landmarker
        landmarker = _get_landmarker()
        cnn_model = tf.keras.models.load_model(model_path)

        visualize_hybrid(cnn_model, landmarker, args.image,
                         norm_stats=norm_stats, output_dir=args.output)
        landmarker.close()
