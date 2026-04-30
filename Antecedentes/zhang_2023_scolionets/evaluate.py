"""
Evaluación K-Fold — Zhang et al. (2023) ScolioNets
====================================================
Evalúa la arquitectura ScolioNets (ResNet50 + ABN) sobre el dataset
de Escalante usando el mismo protocolo que E1/E2/E3:
  - 5-Fold Cross-Validation sobre imágenes originales
  - Train fold: orig + aug_* del grupo train
  - Test fold:  solo orig del grupo test
  - Métricas: Accuracy, F1-macro, AUC-ROC (media ± std)

Uso:
    python evaluate.py
    python evaluate.py --k 5 --epochs_head 10 --epochs_fine 20
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

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import DATASET_AUG_DIR, OUTPUT_DIR, CLASSES, SEED, BATCH_SIZE
from cross_validate import get_original_images, build_fold_paths
from model import build_scolionets_single_output

IMG_SIZE = (224, 224)
LR_HEAD  = 1e-3
LR_FINE  = 1e-5


def _make_ds(paths_arr, labels_arr, shuffle=False, seed=SEED):
    """Construye tf.data.Dataset con preprocesamiento ResNet50."""
    AUTOTUNE = tf.data.AUTOTUNE

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        # Preprocesamiento estándar ResNet50 (camelCase BGR, centrado en ImageNet)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        (paths_arr.tolist(), labels_arr.astype(np.float32).tolist())
    )
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def kfold_zhang2023(dataset_dir, output_dir, k=5, epochs_head=10,
                    epochs_fine=20, seed=SEED):
    """
    K-Fold CV para ScolioNets (Zhang 2023) sobre el dataset de Escalante.

    Fases de entrenamiento (misma estrategia que E1):
        Fase 1: backbone congelado, lr=1e-3, epochs_head épocas
        Fase 2: descongelar conv5_block* (últimas capas ResNet50), lr=1e-5
    """
    out = os.path.join(output_dir, "zhang_2023_scolionets")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*60)
    print("  K-FOLD CV — Zhang et al. (2023) ScolioNets")
    print("  ResNet50 + Attention Branch Network (binario)")
    print("█"*60)

    orig_paths, orig_labels = get_original_images(dataset_dir)
    print(f"  Total originales: {len(orig_paths)}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(orig_paths, orig_labels), 1):

        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────────────")

        train_orig = orig_paths[train_idx]
        train_lbl  = orig_labels[train_idx]
        test_orig  = orig_paths[test_idx]
        test_lbl   = orig_labels[test_idx]

        train_all, train_all_lbl = build_fold_paths(
            train_orig, train_lbl, dataset_dir
        )
        print(f"    Train: {len(train_all)} | Test: {len(test_orig)}")

        train_ds = _make_ds(train_all, train_all_lbl, shuffle=True, seed=seed)
        test_ds  = _make_ds(test_orig, test_lbl)

        # ── Fase 1: cabeza solamente ──────────────────────────────
        model = build_scolionets_single_output(trainable_base=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_HEAD),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_ds, epochs=epochs_head, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=4, restore_best_weights=True)]
        )

        # ── Fase 2: fine-tuning de las últimas capas ResNet50 ─────
        backbone = model.get_layer("resnet50")
        backbone.trainable = True
        for layer in backbone.layers:
            # Descongelar solo conv5_block (equivalente a block6/7 en EfficientNet)
            if not layer.name.startswith("conv5_block"):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_FINE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_ds, epochs=epochs_fine, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=4, restore_best_weights=True)]
        )

        # ── Evaluación ────────────────────────────────────────────
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
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

        fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})
        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        tf.keras.backend.clear_session()

    _report_and_plot(fold_metrics, out, k)
    return fold_metrics


def _report_and_plot(fold_metrics, output_dir, k):
    accs = [m["acc"] for m in fold_metrics]
    f1s  = [m["f1"]  for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]

    print(f"\n  {'═'*55}")
    print(f"  RESULTADOS — Zhang et al. (2023) ScolioNets  ({k}-Fold CV)")
    print(f"  {'═'*55}")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC-ROC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  {'═'*55}")

    with open(os.path.join(output_dir, "kfold_metrics.pkl"), "wb") as f:
        pickle.dump(fold_metrics, f)

    # Gráfica de barras por fold
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    folds  = [f"Fold {i+1}" for i in range(k)]
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    for ax, values, title in zip(
        axes,
        [accs, f1s, [m["auc"] for m in fold_metrics]],
        ["Accuracy", "F1-score (macro)", "AUC-ROC"]
    ):
        bars = ax.bar(folds, values, color=colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}" if not np.isnan(v) else "N/A",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        mean_v = np.nanmean(values)
        ax.axhline(mean_v, color="crimson", linewidth=1.5, linestyle="--",
                   label=f"Media: {mean_v:.3f}")
        ax.fill_between(range(k),
                        mean_v - np.nanstd(values),
                        mean_v + np.nanstd(values),
                        alpha=0.12, color="crimson",
                        label=f"±1σ: {np.nanstd(values):.3f}")

        ax.set_ylim(max(0, np.nanmin(values) - 0.1), 1.08)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"5-Fold CV — Zhang et al. (2023) ScolioNets\n"
        f"ResNet50 + Attention Branch Network  |  "
        f"Acc {np.mean(accs):.3f}±{np.std(accs):.3f}  "
        f"F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
        f"AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "kfold_zhang2023.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica guardada: {path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--epochs_head", type=int, default=10)
    parser.add_argument("--epochs_fine", type=int, default=20)
    parser.add_argument("--dataset",     default=DATASET_AUG_DIR)
    parser.add_argument("--output",      default=os.path.join(OUTPUT_DIR, "antecedentes"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    kfold_zhang2023(
        dataset_dir=args.dataset,
        output_dir=args.output,
        k=args.k,
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
    )
