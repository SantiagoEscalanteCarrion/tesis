"""
Evaluación K-Fold + Robustez — Duan et al. (2025) Dual AttentionUNet
=====================================================================
Entrena el modelo UNA vez por fold y produce dos evaluaciones:

  1. K-Fold CV limpio  → Accuracy / F1 / AUC sin ruido
  2. Robustez al ruido → curva accuracy vs. σ (ruido de píxeles)
     σ ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.50} × 255

El ruido se aplica sobre el test set después de entrenar, por lo que
no hay reentrenamiento extra.

Uso:
    python evaluate.py
    python evaluate.py --k 5 --epochs 20
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import DATASET_AUG_DIR, OUTPUT_DIR, CLASSES, SEED, BATCH_SIZE
from cross_validate import get_original_images, build_fold_paths
from model import build_dual_attention_unet_classifier

IMG_SIZE     = (256, 256)
LR           = 1e-3
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

def _make_ds(paths_arr, labels_arr, pixel_sigma=0.0, shuffle=False, seed=SEED):
    """
    Dataset con normalización simple [0, 255] → manejo interno por Rescaling.
    pixel_sigma > 0 aplica ruido gaussiano antes de la normalización: N(0, σ×255).
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        if pixel_sigma > 0.0:
            noise = tf.random.normal(tf.shape(img), stddev=pixel_sigma * 255.0)
            img = tf.clip_by_value(img + noise, 0.0, 255.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        (paths_arr.tolist(), labels_arr.astype(np.float32).tolist())
    )
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ─────────────────────────────────────────────────────────────
# K-FOLD CV + ROBUSTEZ (un solo loop de entrenamiento)
# ─────────────────────────────────────────────────────────────

def evaluate_duan2025(dataset_dir, output_dir, k=5, epochs=20, seed=SEED):
    """
    Por cada fold:
      1. Entrena Dual AttentionUNet (una sola fase, sin pretrained backbone)
      2. Evalúa en test limpio  → métricas K-Fold
      3. Evalúa en test ruidoso → curva de robustez
    """
    out = os.path.join(output_dir, "duan_2025_dual_attention")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*60)
    print("  Duan et al. (2025) — Dual AttentionUNet")
    print("  Encoder U-Net + SE Channel Attention + Spatial Self-Attention")
    print("  K-Fold CV + Robustez al ruido (un solo entrenamiento)")
    print("█"*60)

    orig_paths, orig_labels = get_original_images(dataset_dir)
    print(f"  Total originales: {len(orig_paths)}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_metrics  = []
    noise_results = {s: [] for s in NOISE_LEVELS}

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(orig_paths, orig_labels), 1):

        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────────────")

        train_orig = orig_paths[train_idx]
        train_lbl  = orig_labels[train_idx]
        test_orig  = orig_paths[test_idx]
        test_lbl   = orig_labels[test_idx]

        train_all, train_all_lbl = build_fold_paths(
            train_orig, train_lbl, dataset_dir)
        print(f"    Train: {len(train_all)} | Test: {len(test_orig)}")

        train_ds = _make_ds(train_all, train_all_lbl, shuffle=True, seed=seed)

        # ── Entrenamiento único (sin backbone preentrenado) ───────────
        print("    Entrenando Dual AttentionUNet...")
        model = build_dual_attention_unet_classifier(dropout=0.5)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True)]
        )

        # ── Evaluar en cada nivel de ruido ────────────────────────────
        for sigma in NOISE_LEVELS:
            test_ds = _make_ds(test_orig, test_lbl, pixel_sigma=sigma)

            y_true, y_prob = [], []
            for imgs, lbls in test_ds:
                probs = model.predict(imgs, verbose=0)
                y_prob.extend(probs.flatten().tolist())
                y_true.extend(lbls.numpy().flatten().tolist())

            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            auc = roc_auc_score(y_true, y_prob) \
                  if len(np.unique(y_true)) > 1 else float("nan")

            noise_results[sigma].append({"acc": acc, "f1": f1, "auc": auc})

            if sigma == 0.0:
                fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})

            print(f"    σ={sigma:.2f}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        tf.keras.backend.clear_session()

    # ── Guardar y graficar ────────────────────────────────────────────
    with open(os.path.join(out, "kfold_metrics.pkl"), "wb") as f:
        pickle.dump(fold_metrics, f)
    with open(os.path.join(out, "robustness_results.pkl"), "wb") as f:
        pickle.dump(noise_results, f)

    _plot_kfold(fold_metrics, out, k)
    _plot_robustness(noise_results, out)

    return fold_metrics, noise_results


# ─────────────────────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────────────────────

def _plot_kfold(fold_metrics, output_dir, k):
    accs = [m["acc"] for m in fold_metrics]
    f1s  = [m["f1"]  for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics]

    print(f"\n  {'═'*55}")
    print(f"  K-FOLD CV (k={k}) — Duan et al. (2025) Dual AttentionUNet")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC-ROC  : {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
    print(f"  {'═'*55}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    folds  = [f"Fold {i+1}" for i in range(k)]
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    for ax, values, title in zip(axes,
                                  [accs, f1s, aucs],
                                  ["Accuracy", "F1-score (macro)", "AUC-ROC"]):
        bars = ax.bar(folds, values, color=colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}" if not np.isnan(v) else "N/A",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        mean_v = np.nanmean(values)
        ax.axhline(mean_v, color="crimson", linewidth=1.5, linestyle="--",
                   label=f"Media: {mean_v:.3f}")
        ax.fill_between(range(k),
                        mean_v - np.nanstd(values), mean_v + np.nanstd(values),
                        alpha=0.12, color="crimson",
                        label=f"±1σ: {np.nanstd(values):.3f}")
        ax.set_ylim(max(0, np.nanmin(values) - 0.1), 1.08)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"5-Fold CV — Duan et al. (2025) Dual AttentionUNet\n"
        f"Acc {np.mean(accs):.3f}±{np.std(accs):.3f}  "
        f"F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
        f"AUC {np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "kfold_duan2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica K-Fold guardada: {path}")
    plt.show()


def _plot_robustness(noise_results, output_dir):
    mean_accs = [np.mean([m["acc"] for m in noise_results[s]]) for s in NOISE_LEVELS]
    std_accs  = [np.std( [m["acc"] for m in noise_results[s]]) for s in NOISE_LEVELS]
    mean_f1s  = [np.mean([m["f1"]  for m in noise_results[s]]) for s in NOISE_LEVELS]
    std_f1s   = [np.std( [m["f1"]  for m in noise_results[s]]) for s in NOISE_LEVELS]
    mean_aucs = [np.nanmean([m["auc"] for m in noise_results[s]]) for s in NOISE_LEVELS]
    std_aucs  = [np.nanstd( [m["auc"] for m in noise_results[s]]) for s in NOISE_LEVELS]

    print(f"\n  {'═'*55}")
    print(f"  ROBUSTEZ — Duan et al. (2025) Dual AttentionUNet")
    print(f"  {'σ':<8} {'Accuracy':>12} {'F1':>12} {'AUC':>12}")
    print(f"  {'─'*46}")
    for s, a, sa, f, sf, u, su in zip(NOISE_LEVELS,
                                       mean_accs, std_accs,
                                       mean_f1s,  std_f1s,
                                       mean_aucs, std_aucs):
        print(f"  {s:<8.2f} {a:>7.4f}±{sa:.3f} {f:>7.4f}±{sf:.3f} {u:>7.4f}±{su:.3f}")
    print(f"  {'═'*55}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")

    for ax, means, stds, title in zip(
        axes,
        [mean_accs, mean_f1s, mean_aucs],
        [std_accs,  std_f1s,  std_aucs],
        ["Accuracy", "F1-score (macro)", "AUC-ROC"]
    ):
        means = np.array(means); stds = np.array(stds)
        ax.plot(NOISE_LEVELS, means, "o-", color="#2563eb",
                linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(NOISE_LEVELS, means - stds, means + stds,
                        alpha=0.15, color="#2563eb")
        for x, m in zip(NOISE_LEVELS, means):
            ax.annotate(f"{m:.3f}", (x, m),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Nivel de ruido (σ × std)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(max(0, np.nanmin(means) - 0.15), 1.08)
        ax.set_xticks(NOISE_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"Robustez bajo Ruido Gaussiano — Duan et al. (2025) Dual AttentionUNet\n"
        f"Accuracy limpia: {mean_accs[0]:.3f} → máximo ruido (σ=0.5): {mean_accs[-1]:.3f}",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_duan2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica robustez guardada: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=5)
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--dataset", default=DATASET_AUG_DIR)
    parser.add_argument("--output",
                        default=os.path.join(OUTPUT_DIR, "antecedentes"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    evaluate_duan2025(
        dataset_dir=args.dataset,
        output_dir=args.output,
        k=args.k,
        epochs=args.epochs,
    )
