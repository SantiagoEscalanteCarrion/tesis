"""
ablation_noise_e3.py — Ablation study del modelo híbrido E3 bajo ruido Gaussiano
==================================================================================
Cuantifica si el colapso de E3 bajo ruido se debe a la rama CNN, a la rama de
pose, o a ambas, evaluando tres condiciones con el modelo pre-entrenado:

  A (both):      ruido aplicado a rama CNN (píxeles) y rama pose (features)
  B (cnn_only):  ruido solo en rama CNN; la rama pose recibe features limpias
  C (pose_only): ruido solo en rama pose; la rama CNN recibe imagen limpia

Metodología: modelo fijo (hybrid_final.keras), test split fijo (seed=42),
5 seeds de ruido por nivel para estimar varianza. No se reentrenan pesos.
Niveles: σ ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.50} — idénticos a Table V.

Outputs en outputs/robustness/robustness_ablation/:
  results_both.pkl, results_cnn_only.pkl, results_pose_only.pkl
  ablation_table.csv
  ablation_figure.png

Ejecutar desde la raíz del proyecto:
  python ablation_noise_e3.py
"""

import os
import sys
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DATASET_AUG_DIR, OUTPUT_DIR, IMG_SIZE
from data_utils import grouped_split
from model_io import load_model_compat

# ── Constantes ─────────────────────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
N_SEEDS      = 5
BASE_SEED    = 42
OUT_DIR      = Path(OUTPUT_DIR) / "robustness" / "robustness_ablation"

COND_LABELS = {
    "both":      "A — CNN + Pose (both branches)",
    "cnn_only":  "B — CNN branch only",
    "pose_only": "C — Pose branch only",
}
COND_COLORS = {
    "both":      "#7c3aed",
    "cnn_only":  "#2563eb",
    "pose_only": "#ea580c",
}


# ── Utilidades de preprocesamiento ─────────────────────────────────────────

def _preprocess_image(img_path, pixel_sigma, rng):
    """Carga, corrompe (si sigma>0) y preprocesa imagen para EfficientNetB0.

    Orden igual al de noise_robustness.py: resize → add noise → clip → preprocess.
    Devuelve array (1, H, W, 3) float32 listo para model.predict.
    """
    import tensorflow as tf
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")
    img = cv2.resize(img, IMG_SIZE).astype(np.float32)
    if pixel_sigma > 0.0:
        noise = rng.normal(0.0, pixel_sigma * 255.0, img.shape).astype(np.float32)
        img   = np.clip(img + noise, 0.0, 255.0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, 0)          # (1, 224, 224, 3)


def _evaluate_condition(model, test_paths, test_labels, test_feats_raw,
                        norm_stats, sigma, noise_mode, seed):
    """Evalúa E3 con una condición y seed dados; retorna dict {acc, f1, auc}."""
    rng  = np.random.default_rng(seed)
    mean = norm_stats["mean"].astype(np.float32)
    std  = (norm_stats["std"] + 1e-8).astype(np.float32)
    feats_norm = (test_feats_raw - mean) / std   # (N, 12)

    y_true, y_prob = [], []

    for i, (path, label) in enumerate(zip(test_paths, test_labels)):
        # Saltar imágenes donde la detección de pose falló (vector de ceros)
        if np.all(test_feats_raw[i] == 0.0):
            continue

        # ── Imagen ────────────────────────────────────────────────────
        p_sigma = sigma if noise_mode in ("cnn_only", "both") else 0.0
        img_arr = _preprocess_image(path, p_sigma, rng)

        # ── Features de pose ──────────────────────────────────────────
        feats = feats_norm[i].copy()
        if sigma > 0.0 and noise_mode in ("pose_only", "both"):
            # Ruido en espacio normalizado: escala sigma ≈ 1 std normalizado
            feats = feats + rng.normal(0.0, sigma, feats.shape).astype(np.float32)

        prob = float(
            model.predict([img_arr, feats[np.newaxis]], verbose=0)[0, 0]
        )
        y_true.append(int(label))
        y_prob.append(prob)

    if len(y_true) < 2:
        return {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}

    y_pred = [int(p >= 0.5) for p in y_prob]
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


# ── Loop principal ─────────────────────────────────────────────────────────

def run_ablation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Modelo ────────────────────────────────────────────────────────────
    model_path = Path(OUTPUT_DIR) / "hybrid" / "hybrid_final.keras"
    print(f"[ablation] Cargando {model_path} ...")
    model = load_model_compat(str(model_path))

    # ── Test set (mismo split fijo que usó E3 en evaluación principal) ───
    split       = grouped_split(DATASET_AUG_DIR)
    test_split  = split["test"]
    test_paths  = [p for p, _ in test_split]
    test_labels = [l for _, l in test_split]

    # ── Features pre-extraídas del test set ──────────────────────────────
    feats_pkl = Path(OUTPUT_DIR) / "pose" / "split_features.pkl"
    with open(feats_pkl, "rb") as f:
        split_feats = pickle.load(f)
    test_feats_raw = split_feats["test"]["X"].astype(np.float32)
    test_feats_y   = split_feats["test"]["y"]

    # Verificar alineamiento de etiquetas (advertir si difieren)
    n = min(len(test_paths), len(test_feats_raw))
    if n < len(test_paths):
        print(f"[ablation] AVISO: test set tiene {len(test_paths)} imágenes pero "
              f"split_features.pkl contiene {len(test_feats_raw)} muestras — "
              f"usando las primeras {n}.")
    test_paths     = test_paths[:n]
    test_labels    = test_labels[:n]
    test_feats_raw = test_feats_raw[:n]
    test_feats_y   = test_feats_y[:n]

    label_match = np.array(test_labels) == test_feats_y
    if not label_match.all():
        n_mismatch = (~label_match).sum()
        print(f"[ablation] AVISO: {n_mismatch} etiquetas no coinciden entre "
              f"grouped_split() y split_features.pkl. Revisar alineamiento.")
    else:
        print(f"[ablation] Alineamiento OK — {n} imágenes de test.")

    # ── Norm stats del train de E3 ────────────────────────────────────────
    norm_path = Path(OUTPUT_DIR) / "hybrid" / "pose_norm_stats.pkl"
    with open(norm_path, "rb") as f:
        norm_stats = pickle.load(f)

    n_pose_ok = int(np.any(test_feats_raw != 0, axis=1).sum())
    print(f"[ablation] Imágenes con pose detectada: {n_pose_ok}/{n}")

    # ── Evaluación por condición ──────────────────────────────────────────
    conditions  = ("both", "cnn_only", "pose_only")
    all_results = {}

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"  Condición: {COND_LABELS[cond]}")
        print(f"{'='*60}")
        results = {}
        for sigma in NOISE_LEVELS:
            seed_results = []
            for s in range(N_SEEDS):
                seed = BASE_SEED + s * 100
                r    = _evaluate_condition(
                    model, test_paths, test_labels, test_feats_raw,
                    norm_stats, sigma, cond, seed
                )
                seed_results.append(r)
            results[sigma] = seed_results
            m_acc = np.nanmean([r["acc"] for r in seed_results])
            m_f1  = np.nanmean([r["f1"]  for r in seed_results])
            m_auc = np.nanmean([r["auc"] for r in seed_results])
            print(f"  σ={sigma:.2f} → Acc={m_acc:.3f}  F1={m_f1:.3f}  AUC={m_auc:.3f}")
        all_results[cond] = results

        pkl_file = OUT_DIR / f"results_{cond}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(results, f)
        print(f"  → {pkl_file}")

    _save_csv(all_results)
    _plot_figure(all_results)
    _print_table(all_results)
    return all_results


# ── Outputs ────────────────────────────────────────────────────────────────

def _save_csv(all_results):
    csv_path = OUT_DIR / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "sigma",
                         "acc_mean", "acc_std",
                         "f1_mean",  "f1_std",
                         "auc_mean", "auc_std"])
        for cond, results in all_results.items():
            for sigma in NOISE_LEVELS:
                seeds = results[sigma]
                accs  = [r["acc"] for r in seeds if not np.isnan(r["acc"])]
                f1s   = [r["f1"]  for r in seeds if not np.isnan(r["f1"])]
                aucs  = [r["auc"] for r in seeds if not np.isnan(r["auc"])]
                writer.writerow([
                    cond, f"{sigma:.2f}",
                    f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}",
                    f"{np.mean(f1s):.4f}",  f"{np.std(f1s):.4f}",
                    f"{np.mean(aucs):.4f}" if aucs else "nan",
                    f"{np.std(aucs):.4f}"  if aucs else "nan",
                ])
    print(f"\n[ablation] CSV → {csv_path}")


def _plot_figure(all_results):
    """Figura con 3 subplots (Acc/F1/AUC) y 3 curvas (A/B/C) con bandas ±1σ."""
    metrics = [("acc", "Accuracy"), ("f1", "F1-score (macro)"), ("auc", "AUC-ROC")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        ax.set_facecolor("#fafbfc")
        for cond in ("both", "cnn_only", "pose_only"):
            results = all_results[cond]
            sigmas  = sorted(results.keys())
            means, stds = [], []
            for sigma in sigmas:
                vals = [r[metric_key] for r in results[sigma]
                        if not np.isnan(r[metric_key])]
                means.append(np.mean(vals) if vals else float("nan"))
                stds.append(np.std(vals)   if vals else float("nan"))
            means = np.array(means)
            stds  = np.array(stds)
            color = COND_COLORS[cond]
            label = COND_LABELS[cond]
            ax.plot(sigmas, means, "o-", color=color, label=label,
                    linewidth=2.5, markersize=7,
                    markerfacecolor="white", markeredgewidth=2)
            ax.fill_between(sigmas, means - stds, means + stds,
                            color=color, alpha=0.12)

        ax.set_xlabel("Noise level σ", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(NOISE_LEVELS)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower left")

    fig.suptitle(
        "E3 Hybrid — Ablation: Branch-wise Noise Injection\n"
        "(fixed pre-trained weights, 5 noise seeds per level)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out_path = OUT_DIR / "ablation_figure.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Figura → {out_path}")


def _print_table(all_results):
    col_w = 32
    header = (f"{'Condition':<{col_w}} {'σ':>5}  "
              f"{'Accuracy':>13}  {'F1-macro':>13}  {'AUC-ROC':>13}")
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for cond in ("both", "cnn_only", "pose_only"):
        results = all_results[cond]
        label   = COND_LABELS[cond]
        for sigma in NOISE_LEVELS:
            seeds = results[sigma]
            accs  = [r["acc"] for r in seeds if not np.isnan(r["acc"])]
            f1s   = [r["f1"]  for r in seeds if not np.isnan(r["f1"])]
            aucs  = [r["auc"] for r in seeds if not np.isnan(r["auc"])]
            print(f"{label:<{col_w}} {sigma:>5.2f}  "
                  f"{np.mean(accs):>5.3f}±{np.std(accs):.3f}  "
                  f"{np.mean(f1s):>5.3f}±{np.std(f1s):.3f}  "
                  f"{(np.mean(aucs) if aucs else float('nan')):>5.3f}"
                  f"±{(np.std(aucs) if aucs else float('nan')):.3f}")
        print()
    print(sep)


if __name__ == "__main__":
    run_ablation()
