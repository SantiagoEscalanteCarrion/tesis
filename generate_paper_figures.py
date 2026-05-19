"""
generate_paper_figures.py — Figuras para el paper de tesis
===========================================================
Genera los tres outputs visuales necesarios para el paper:

  A) Robustez combinada en inglés (E1 CNN + E2 Pose + E3 Hybrid en un gráfico)
  B) Grad-CAM panel para E1: 2 imágenes side-by-side con heatmap superpuesto
  C) SHAP bar chart para E2: 12 features ordenadas por importancia media

No requiere reentrenar ningún modelo. Usa archivos ya guardados en outputs/.

Requisito extra:
    pip install shap

Uso:
    python generate_paper_figures.py --all \\
        --img_yes dataset_augmented/scoliosis_yes/orig_0.jpg \\
        --img_no  dataset_augmented/scoliosis_no/orig_0.jpg

    python generate_paper_figures.py --robustness
    python generate_paper_figures.py --gradcam --img_yes <ruta> --img_no <ruta>
    python generate_paper_figures.py --shap
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR, POSE_FEATURE_NAMES, IMG_SIZE, SEED

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

FEATURE_NAMES_EN = [
    "Shoulder Height Diff.",
    "Hip Height Diff.",
    "Elbow Height Diff.",
    "Wrist Height Diff.",
    "Shoulder Width",
    "Hip Width",
    "Shoulder Asymmetry Ratio",
    "Hip Asymmetry Ratio",
    "Trunk Lateral Tilt",
    "Trunk Tilt Angle",
    "Spine Lateral Deviation",
    "Body Height Proxy",
]

MODEL_COLORS = {
    "E1 CNN":    "#2563eb",
    "E2 Pose":   "#16a34a",
    "E3 Hybrid": "#dc2626",
}


# ─────────────────────────────────────────────────────────────
# A) ROBUSTEZ COMBINADA EN INGLÉS
# ─────────────────────────────────────────────────────────────

def plot_robustness_combined_en(pkl_cnn, pkl_pose, pkl_hybrid, output_dir):
    """
    Carga los pkl de robustez de E1, E2 y E3 y genera una figura combinada en inglés.
    """
    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _stats(results, key):
        means = [np.mean([m[key] for m in results[s]]) for s in NOISE_LEVELS]
        stds  = [np.std( [m[key] for m in results[s]]) for s in NOISE_LEVELS]
        return np.array(means), np.array(stds)

    data = {
        "E1 CNN":    _load(pkl_cnn),
        "E2 Pose":   _load(pkl_pose),
        "E3 Hybrid": _load(pkl_hybrid),
    }

    metrics = [
        ("acc", "Accuracy"),
        ("f1",  "F1-score (macro)"),
        ("auc", "AUC-ROC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")

    for ax, (key, title) in zip(axes, metrics):
        for model_name, results in data.items():
            if key == "auc":
                means = [np.nanmean([m[key] for m in results[s]]) for s in NOISE_LEVELS]
                stds  = [np.nanstd( [m[key] for m in results[s]]) for s in NOISE_LEVELS]
                means, stds = np.array(means), np.array(stds)
            else:
                means, stds = _stats(results, key)

            color = MODEL_COLORS[model_name]
            ax.plot(NOISE_LEVELS, means, "o-", color=color, linewidth=2.5,
                    markersize=7, markerfacecolor="white", markeredgewidth=2,
                    label=model_name)
            ax.fill_between(NOISE_LEVELS, means - stds, means + stds,
                            alpha=0.12, color=color)

        ax.set_xlabel("Noise level (σ × std)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(NOISE_LEVELS)
        ax.set_ylim(0, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        "Robustness under Gaussian Noise — E1 CNN · E2 Pose · E3 Hybrid",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "robustness_combined_en.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  [A] Robustness combined (EN) saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# B) GRAD-CAM PANEL (E1 CNN)
# ─────────────────────────────────────────────────────────────

def plot_gradcam_panel(model_path, img_yes_path, img_no_path, output_dir):
    """
    Genera panel 2×2: (scoliosis_yes, scoliosis_no) × (original, Grad-CAM++ overlay).
    Usa tf-keras-vis GradcamPlusPlus para manejar el submodelo EfficientNetB0 anidado.
    """
    import tensorflow as tf
    from matplotlib import cm
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    model = tf.keras.models.load_model(model_path)

    # Reemplaza la activación final (sigmoid) por lineal para que los gradientes fluyan
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)

    def _score_scoliosis(output):
        return output[:, 0]

    def _gradcam_overlay(image_path):
        img_raw = cv2.imread(image_path)
        img_raw = cv2.resize(img_raw, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        img_arr = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_arr = tf.keras.preprocessing.image.img_to_array(img_arr)
        img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)  # (1, H, W, 3)

        # Predicción original (sin modifier)
        prob = float(model.predict(img_arr, verbose=0)[0, 0])

        # Grad-CAM++ — penultimate_layer=-1 busca automáticamente la última conv
        cam = gradcam(_score_scoliosis, img_arr, penultimate_layer=-1)
        heatmap = cam[0]  # (H, W) ya normalizado [0, 1]

        heatmap_r = cv2.resize(heatmap, IMG_SIZE)
        colored   = (cm.jet(heatmap_r)[:, :, :3] * 255).astype(np.uint8)
        overlay   = cv2.addWeighted(img_raw, 0.55, colored, 0.45, 0)
        overlay   = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return img_rgb, overlay, prob

    yes_orig, yes_overlay, yes_prob = _gradcam_overlay(img_yes_path)
    no_orig,  no_overlay,  no_prob  = _gradcam_overlay(img_no_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_facecolor("#f8f9fa")

    axes[0, 0].imshow(yes_orig);    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")

    axes[0, 1].imshow(yes_overlay); axes[0, 1].axis("off")
    axes[0, 1].set_title(
        f"Grad-CAM Overlay\nPred: Scoliosis ({yes_prob:.2f})",
        fontsize=12, fontweight="bold", color="#dc2626"
    )

    axes[1, 0].imshow(no_orig);    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original Image", fontsize=12, fontweight="bold")

    axes[1, 1].imshow(no_overlay); axes[1, 1].axis("off")
    axes[1, 1].set_title(
        f"Grad-CAM Overlay\nPred: Healthy ({1 - no_prob:.2f})",
        fontsize=12, fontweight="bold", color="#16a34a"
    )

    # Etiquetas de fila
    for ax, label, color in zip(
        [axes[0, 0], axes[1, 0]],
        ["Scoliosis (Positive)", "Healthy (Negative)"],
        ["#dc2626", "#16a34a"]
    ):
        ax.set_ylabel(label, fontsize=13, fontweight="bold",
                      color=color, labelpad=10)

    fig.suptitle("Grad-CAM Explanations — E1 EfficientNetB0 CNN",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gradcam_panel.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  [B] Grad-CAM panel saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# C) SHAP BAR CHART (E2 Pose)
# ─────────────────────────────────────────────────────────────

def plot_shap_barchart(model_path, features_cache_path, output_dir):
    """
    Genera bar chart horizontal de importancia media SHAP para el modelo E2 (RandomForest).
    """
    import shap

    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    with open(features_cache_path, "rb") as f:
        cache = pickle.load(f)

    X = cache["X"]
    scaler = pipe.named_steps["scaler"]
    rf     = pipe.named_steps["clf"]
    X_scaled = scaler.transform(X)

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # Para clasificación binaria: shap_values es lista [class0, class1] o array 3D
    if isinstance(shap_values, list):
        sv = shap_values[1]   # clase positiva (scoliosis_yes)
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    mean_abs = np.mean(np.abs(sv), axis=0)
    order    = np.argsort(mean_abs)  # ascendente para barh (más importante arriba)

    names_en = [FEATURE_NAMES_EN[i] for i in order]
    values   = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#fafbfc")

    colors = ["#2563eb" if v == values.max() else "#93c5fd" for v in values]
    bars = ax.barh(names_en, values, color=colors, edgecolor="white", height=0.65)

    for bar, v in zip(bars, values):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance — E2 Pose (RandomForest + SHAP)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "shap_barchart.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  [C] SHAP bar chart saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera figuras del paper: robustez combinada, Grad-CAM, SHAP"
    )

    parser.add_argument("--all",       action="store_true", help="Generar las 3 figuras")
    parser.add_argument("--robustness",action="store_true", help="Figura A: robustez combinada (EN)")
    parser.add_argument("--gradcam",   action="store_true", help="Figura B: Grad-CAM panel")
    parser.add_argument("--shap",      action="store_true", help="Figura C: SHAP bar chart")

    # Rutas de pkl de robustez
    parser.add_argument("--pkl_cnn",    default=os.path.join(OUTPUT_DIR, "robustness", "robustness_cnn",    "robustness_results.pkl"))
    parser.add_argument("--pkl_pose",   default=os.path.join(OUTPUT_DIR, "robustness", "robustness_pose",   "robustness_results.pkl"))
    parser.add_argument("--pkl_hybrid", default=os.path.join(OUTPUT_DIR, "robustness", "robustness_hybrid", "robustness_results.pkl"))

    # Rutas de modelos guardados
    parser.add_argument("--cnn_model",  default=os.path.join(OUTPUT_DIR, "cnn",  "cnn_final.keras"))
    parser.add_argument("--pose_model", default=os.path.join(OUTPUT_DIR, "pose", "pose_RandomForest.pkl"))
    parser.add_argument("--features",   default=os.path.join(OUTPUT_DIR, "feature_diagnosis", "orig_features_cache.pkl"))

    # Imágenes para Grad-CAM
    parser.add_argument("--img_yes", default=None, help="Ruta a imagen scoliosis_yes (orig_*.jpg)")
    parser.add_argument("--img_no",  default=None, help="Ruta a imagen scoliosis_no (orig_*.jpg)")

    # Directorio de salida
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "paper_figures"))

    args = parser.parse_args()
    run_all        = args.all
    run_robustness = args.robustness or run_all
    run_gradcam    = args.gradcam    or run_all
    run_shap       = args.shap       or run_all

    print("\n" + "█"*55)
    print("  Generando figuras para el paper de tesis")
    print("█"*55)

    if run_robustness:
        print("\n[A] Robustez combinada en inglés...")
        plot_robustness_combined_en(
            pkl_cnn=args.pkl_cnn,
            pkl_pose=args.pkl_pose,
            pkl_hybrid=args.pkl_hybrid,
            output_dir=args.output,
        )

    if run_gradcam:
        if not args.img_yes or not args.img_no:
            print("\n[B] Grad-CAM: falta --img_yes y --img_no. Omitiendo.")
        else:
            print("\n[B] Grad-CAM panel...")
            plot_gradcam_panel(
                model_path=args.cnn_model,
                img_yes_path=args.img_yes,
                img_no_path=args.img_no,
                output_dir=args.output,
            )

    if run_shap:
        print("\n[C] SHAP bar chart...")
        plot_shap_barchart(
            model_path=args.pose_model,
            features_cache_path=args.features,
            output_dir=args.output,
        )

    print(f"\n  Figuras guardadas en: {args.output}")
