"""
cross_validate.py — Evaluación con K-Fold Cross-Validation
============================================================
Estándar gold para datasets médicos pequeños.

Estrategia:
  - 5-fold estratificado sobre las imágenes ORIGINALES (orig_*.jpg)
  - Train fold: originals del fold train + sus aug_* correspondientes
  - Test fold:  SOLO originals del fold de test (sin aumentadas)
  - Se reporta media ± desviación estándar por cada métrica

Esto es lo que piden revisores y comités de tesis para datasets pequeños
de imágenes médicas.

Uso:
    python cross_validate.py --model pose    (rápido, ~5-10 min)
    python cross_validate.py --model cnn     (lento, ~30-60 min)
    python cross_validate.py --model all     (entrena los tres)
"""

import os
import re
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import seaborn as sns

from config import DATASET_AUG_DIR, OUTPUT_DIR, CLASSES, SEED, BATCH_SIZE, IMG_SIZE


# ─────────────────────────────────────────────────────────────
# UTILIDADES DE SPLIT POR FOLD
# ─────────────────────────────────────────────────────────────

def get_original_images(dataset_dir):
    """
    Retorna (paths, labels) de todas las imágenes orig_*.jpg del dataset.
    """
    paths, labels = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.startswith("orig_") and fname.lower().endswith(".jpg"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    return np.array(paths), np.array(labels)


def get_aug_for_orig(orig_path, n_orig_in_class, dataset_dir, class_name):
    """
    Dado un orig_XXXX.jpg, retorna todos los aug_YYYYY.jpg que lo tienen
    como fuente (aug_idx % n_orig == orig_idx).
    """
    fname = os.path.basename(orig_path)
    m = re.search(r"orig_(\d+)", fname)
    if not m:
        return []
    orig_idx = int(m.group(1))

    class_dir = os.path.join(dataset_dir, class_name)
    aug_paths = []
    for f in os.listdir(class_dir):
        if not f.startswith("aug_"):
            continue
        ma = re.search(r"aug_(\d+)", f)
        if not ma:
            continue
        if int(ma.group(1)) % n_orig_in_class == orig_idx:
            aug_paths.append(os.path.join(class_dir, f))
    return aug_paths


def build_fold_paths(train_orig_paths, train_orig_labels, dataset_dir):
    """
    Dado un conjunto de rutas de originales de train, añade sus aug_*.
    Retorna (all_train_paths, all_train_labels).
    """
    # Contar originales por clase para calcular el módulo
    n_orig_per_class = {}
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        n_orig_per_class[label_idx] = len([
            f for f in os.listdir(class_dir)
            if f.startswith("orig_") and f.lower().endswith(".jpg")
        ])

    all_paths = list(train_orig_paths)
    all_labels = list(train_orig_labels)

    for orig_path, label in zip(train_orig_paths, train_orig_labels):
        class_name = CLASSES[label]
        n_orig = n_orig_per_class[label]
        augs = get_aug_for_orig(orig_path, n_orig, dataset_dir, class_name)
        all_paths.extend(augs)
        all_labels.extend([label] * len(augs))

    return np.array(all_paths), np.array(all_labels)


# ─────────────────────────────────────────────────────────────
# K-FOLD: MODELO POSE
# ─────────────────────────────────────────────────────────────

def kfold_pose(dataset_dir, output_dir, k=5, seed=SEED):
    """
    K-Fold CV para el modelo Pose + RandomForest.
    Rápido (~5-10 min total).
    """
    from model_pose import extract_pose_features_from_image, _get_landmarker
    from model_pose import build_classifiers

    out = os.path.join(output_dir, "kfold_pose")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*55)
    print(f"  K-FOLD CV (k={k}) — Pose + ML")
    print("█"*55)

    orig_paths, orig_labels = get_original_images(dataset_dir)
    print(f"  Total imágenes originales: {len(orig_paths)}")
    print(f"  Distribución: {dict(zip(*np.unique(orig_labels, return_counts=True)))}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics = []

    landmarker = _get_landmarker()

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(orig_paths, orig_labels), 1):
        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────")

        train_orig = orig_paths[train_idx]
        train_lbl  = orig_labels[train_idx]
        test_orig  = orig_paths[test_idx]
        test_lbl   = orig_labels[test_idx]

        # Añadir augmentadas al train
        train_all, train_all_lbl = build_fold_paths(
            train_orig, train_lbl, dataset_dir
        )
        print(f"    Train: {len(train_all)} imgs (orig={len(train_orig)}, aug={len(train_all)-len(train_orig)})")
        print(f"    Test:  {len(test_orig)} imgs (solo orig)")

        # Extraer features de train
        X_train, y_train = [], []
        for path, lbl in zip(train_all, train_all_lbl):
            f = extract_pose_features_from_image(str(path), landmarker)
            if f is not None:
                X_train.append(f); y_train.append(lbl)

        # Extraer features de test
        X_test, y_test = [], []
        for path, lbl in zip(test_orig, test_lbl):
            f = extract_pose_features_from_image(str(path), landmarker)
            if f is not None:
                X_test.append(f); y_test.append(lbl)

        X_train = np.array(X_train, dtype=np.float32)
        X_test  = np.array(X_test,  dtype=np.float32)
        y_train = np.array(y_train)
        y_test  = np.array(y_test)

        # Entrenar RandomForest (el más rápido)
        clf = build_classifiers(seed)["RandomForest"]
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')

        fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})
        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    landmarker.close()

    _report_and_plot(fold_metrics, "Pose + RandomForest", out, k)
    return fold_metrics


# ─────────────────────────────────────────────────────────────
# K-FOLD: MODELO CNN
# ─────────────────────────────────────────────────────────────

def kfold_cnn(dataset_dir, output_dir, k=5, seed=SEED):
    """
    K-Fold CV para el modelo CNN (EfficientNetB0).
    Más lento (~10-20 min por fold según GPU).
    """
    import tensorflow as tf
    from model_cnn import build_cnn_model, get_callbacks
    from config import (EPOCHS_HEAD, EPOCHS_FINE,
                        LEARNING_RATE_HEAD, LEARNING_RATE_FINE)

    out = os.path.join(output_dir, "kfold_cnn")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*55)
    print(f"  K-FOLD CV (k={k}) — CNN EfficientNetB0")
    print("█"*55)

    orig_paths, orig_labels = get_original_images(dataset_dir)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics = []

    AUTOTUNE = tf.data.AUTOTUNE

    def _make_ds(paths_arr, labels_arr, shuffle=False):
        def load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            return img, label
        ds = tf.data.Dataset.from_tensor_slices(
            (paths_arr.tolist(), labels_arr.astype(np.float32).tolist())
        )
        ds = ds.map(load, num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(500, seed=seed)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(orig_paths, orig_labels), 1):
        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────")

        train_orig = orig_paths[train_idx]
        train_lbl  = orig_labels[train_idx]
        test_orig  = orig_paths[test_idx]
        test_lbl   = orig_labels[test_idx]

        train_all, train_all_lbl = build_fold_paths(
            train_orig, train_lbl, dataset_dir
        )
        print(f"    Train: {len(train_all)} | Test: {len(test_orig)}")

        train_ds = _make_ds(train_all, train_all_lbl, shuffle=True)
        test_ds  = _make_ds(test_orig, test_lbl)

        # Fase 1: head training
        model = build_cnn_model(trainable_base=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_HEAD),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(train_ds, epochs=EPOCHS_HEAD, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      patience=4, restore_best_weights=True)])

        # Fase 2: fine-tuning
        bb = model.layers[3]
        bb.trainable = True
        for layer in bb.layers:
            if not layer.name.startswith(("block6", "block7", "top")):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FINE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(train_ds, epochs=EPOCHS_FINE, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      patience=4, restore_best_weights=True)])

        # Evaluar
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
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')

        fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})
        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        tf.keras.backend.clear_session()

    _report_and_plot(fold_metrics, "CNN EfficientNetB0", out, k)
    return fold_metrics


# ─────────────────────────────────────────────────────────────
# REPORTE Y GRÁFICA
# ─────────────────────────────────────────────────────────────

def _report_and_plot(fold_metrics, model_name, output_dir, k):
    accs = [m["acc"] for m in fold_metrics]
    f1s  = [m["f1"]  for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]

    print(f"\n  {'─'*50}")
    print(f"  RESULTADOS FINALES — {model_name}  ({k}-Fold CV)")
    print(f"  {'─'*50}")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1-score : {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"  AUC-ROC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  {'─'*50}")

    # Guardar métricas
    with open(os.path.join(output_dir, "kfold_metrics.pkl"), "wb") as f:
        pickle.dump(fold_metrics, f)

    # ── Gráfica de barras por fold ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    folds = [f"Fold {i+1}" for i in range(k)]
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
        ax.axhline(mean_v, color="crimson", linewidth=1.5,
                   linestyle="--", label=f"Media: {mean_v:.3f}")
        ax.fill_between(range(k), mean_v - np.nanstd(values),
                        mean_v + np.nanstd(values),
                        alpha=0.12, color="crimson",
                        label=f"±1σ: {np.nanstd(values):.3f}")

        ax.set_ylim(max(0, np.nanmin(values) - 0.1), 1.08)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"{k}-Fold Cross-Validation — {model_name}\n"
        f"Accuracy {np.mean(accs):.3f}±{np.std(accs):.3f}  |  "
        f"F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}  |  "
        f"AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"kfold_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica guardada: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation para detección de escoliosis"
    )
    parser.add_argument("--model", choices=["pose", "cnn", "all"],
                        default="pose",
                        help="Modelo a evaluar (default: pose, más rápido)")
    parser.add_argument("--k", type=int, default=5,
                        help="Número de folds (default: 5)")
    parser.add_argument("--dataset", default=DATASET_AUG_DIR)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "crossval"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.model in ("pose", "all"):
        kfold_pose(args.dataset, args.output, k=args.k)

    if args.model in ("cnn", "all"):
        kfold_cnn(args.dataset, args.output, k=args.k)
