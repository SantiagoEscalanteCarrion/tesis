"""
noise_robustness.py — Evaluación de robustez bajo ruido gaussiano
==================================================================
Evalúa cómo degrada el rendimiento de los modelos cuando se introduce
ruido gaussiano progresivo, simulando condiciones clínicas imperfectas:

  - Iluminación variable en consulta
  - Ropa holgada que dificulta la detección de pose
  - Movimiento del paciente durante la toma
  - Imprecisión del estimador MediaPipe en condiciones subóptimas

Para el modelo Pose: ruido en las features geométricas
Para el modelo CNN:  ruido de píxeles (Gaussian pixel noise)

Uso:
    python noise_robustness.py --model pose    (rápido, ~5-10 min)
    python noise_robustness.py --model cnn     (lento, ~20-30 min)
    python noise_robustness.py --model all
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (DATASET_AUG_DIR, OUTPUT_DIR, CLASSES, SEED,
                    POSE_FEATURE_NAMES, BATCH_SIZE, IMG_SIZE)


# Niveles de ruido: fracción de la desviación estándar del training
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


# ─────────────────────────────────────────────────────────────
# UTILIDADES COMUNES
# ─────────────────────────────────────────────────────────────

def get_original_images(dataset_dir):
    """Retorna (paths, labels) de todos los orig_*.jpg."""
    paths, labels = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.startswith("orig_") and fname.lower().endswith(".jpg"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    return np.array(paths), np.array(labels)


def _load_orig_features(dataset_dir, seed=SEED):
    """Carga o extrae features de todos los orig_*.jpg."""
    cache = os.path.join(OUTPUT_DIR, "feature_diagnosis", "orig_features_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["X"], d["y"]

    from model_pose import extract_pose_features_from_image, _get_landmarker
    print("  Extrayendo features de imágenes originales (~2-3 min)...")
    landmarker = _get_landmarker()
    X, y = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if not (fname.startswith("orig_") and fname.lower().endswith(".jpg")):
                continue
            feats = extract_pose_features_from_image(
                os.path.join(class_dir, fname), landmarker)
            if feats is not None:
                X.append(feats); y.append(label_idx)
    landmarker.close()
    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────────────────────────────────────
# ROBUSTEZ — MODELO POSE
# ─────────────────────────────────────────────────────────────

def noise_robustness_pose(dataset_dir, output_dir, k=5, seed=SEED):
    """
    5-fold CV para Pose+RandomForest con niveles crecientes de ruido
    gaussiano en las features (σ = nivel × std_train).
    """
    from model_pose import build_classifiers

    out = os.path.join(output_dir, "robustness_pose")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*60)
    print("  ROBUSTEZ BAJO RUIDO — Pose + RandomForest")
    print("█"*60)

    X, y = _load_orig_features(dataset_dir, seed)
    print(f"  Total: {len(y)} imágenes originales con pose detectada")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # results[noise_level] = list of per-fold dicts
    results = {sigma: [] for sigma in NOISE_LEVELS}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────────────")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Estadísticas del train (para escalar el ruido)
        train_std = X_train.std(axis=0) + 1e-8

        # Entrenar sin ruido (siempre en features limpias)
        clf = build_classifiers(seed)["RandomForest"]
        clf.fit(X_train, y_train)

        for sigma in NOISE_LEVELS:
            if sigma == 0.0:
                X_noisy = X_test
            else:
                noise = np.random.default_rng(seed + fold_idx).normal(
                    loc=0.0,
                    scale=sigma * train_std,
                    size=X_test.shape
                ).astype(np.float32)
                X_noisy = X_test + noise

            y_pred = clf.predict(X_noisy)
            y_prob = clf.predict_proba(X_noisy)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

            results[sigma].append({"acc": acc, "f1": f1, "auc": auc})
            print(f"    σ={sigma:.2f}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    _report_robustness(results, "Pose + RandomForest", out, k)
    return results


# ─────────────────────────────────────────────────────────────
# ROBUSTEZ — MODELO CNN
# ─────────────────────────────────────────────────────────────

def noise_robustness_cnn(dataset_dir, output_dir, k=5, seed=SEED):
    """
    5-fold CV para CNN con niveles crecientes de ruido de píxeles
    (Gaussian pixel noise con σ = nivel × 255).
    """
    import tensorflow as tf
    from model_cnn import build_cnn_model
    from config import EPOCHS_HEAD, EPOCHS_FINE, LEARNING_RATE_HEAD, LEARNING_RATE_FINE
    from cross_validate import build_fold_paths

    out = os.path.join(output_dir, "robustness_cnn")
    os.makedirs(out, exist_ok=True)

    print("\n" + "█"*60)
    print("  ROBUSTEZ BAJO RUIDO — CNN EfficientNetB0")
    print("█"*60)

    orig_paths, orig_labels = get_original_images(dataset_dir)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    AUTOTUNE = tf.data.AUTOTUNE

    def _make_ds(paths_arr, labels_arr, pixel_sigma=0.0, shuffle=False):
        def load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            if pixel_sigma > 0.0:
                noise = tf.random.normal(tf.shape(img), stddev=pixel_sigma * 255.0)
                img = tf.clip_by_value(img + noise, 0.0, 255.0)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            return img, label
        ds = tf.data.Dataset.from_tensor_slices(
            (paths_arr.tolist(), labels_arr.astype(np.float32).tolist())
        )
        ds = ds.map(load, num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(500, seed=seed)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    results = {sigma: [] for sigma in NOISE_LEVELS}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(orig_paths, orig_labels), 1):
        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────────────────")

        train_orig = orig_paths[train_idx]
        train_lbl  = orig_labels[train_idx]
        test_orig  = orig_paths[test_idx]
        test_lbl   = orig_labels[test_idx]

        train_all, train_all_lbl = build_fold_paths(train_orig, train_lbl, dataset_dir)
        train_ds = _make_ds(train_all, train_all_lbl, pixel_sigma=0.0, shuffle=True)

        # Entrenar el modelo (sin ruido)
        model = build_cnn_model(trainable_base=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_HEAD),
                      loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, epochs=EPOCHS_HEAD, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

        bb = model.layers[3]; bb.trainable = True
        for layer in bb.layers:
            if not layer.name.startswith(("block6", "block7", "top")):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FINE),
                      loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, epochs=EPOCHS_FINE, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

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
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

            results[sigma].append({"acc": acc, "f1": f1, "auc": auc})
            print(f"    σ={sigma:.2f}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        tf.keras.backend.clear_session()

    _report_robustness(results, "CNN EfficientNetB0", out, k)
    return results


# ─────────────────────────────────────────────────────────────
# REPORTE Y GRÁFICA
# ─────────────────────────────────────────────────────────────

def _report_robustness(results, model_name, output_dir, k):
    """Imprime tabla y genera gráfica de robustez."""
    print(f"\n  {'═'*60}")
    print(f"  RESUMEN DE ROBUSTEZ — {model_name}")
    print(f"  {'═'*60}")
    print(f"  {'σ (ruido)':<12} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'─'*44}")

    mean_accs, mean_f1s, mean_aucs = [], [], []
    std_accs, std_f1s, std_aucs   = [], [], []

    for sigma in NOISE_LEVELS:
        fold_data = results[sigma]
        accs = [d["acc"] for d in fold_data]
        f1s  = [d["f1"]  for d in fold_data]
        aucs = [d["auc"] for d in fold_data if not np.isnan(d["auc"])]

        m_acc, s_acc = np.mean(accs), np.std(accs)
        m_f1,  s_f1  = np.mean(f1s),  np.std(f1s)
        m_auc, s_auc = (np.mean(aucs), np.std(aucs)) if aucs else (float("nan"), float("nan"))

        mean_accs.append(m_acc); std_accs.append(s_acc)
        mean_f1s.append(m_f1);   std_f1s.append(s_f1)
        mean_aucs.append(m_auc); std_aucs.append(s_auc)

        print(f"  {sigma:<12.2f} {m_acc:>7.4f}±{s_acc:.3f} {m_f1:>7.4f}±{s_f1:.3f} {m_auc:>7.4f}±{s_auc:.3f}")

    print(f"  {'═'*60}")

    # Guardar pkl
    with open(os.path.join(output_dir, "robustness_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # ── Gráfica ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")

    metrics = [
        (mean_accs, std_accs, "Accuracy"),
        (mean_f1s,  std_f1s,  "F1-score (macro)"),
        (mean_aucs, std_aucs, "AUC-ROC"),
    ]
    for ax, (means, stds, title) in zip(axes, metrics):
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(NOISE_LEVELS, means, "o-", color="#2563eb", linewidth=2.5,
                markersize=7, markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(NOISE_LEVELS, means - stds, means + stds,
                        alpha=0.15, color="#2563eb")

        for x, m, s in zip(NOISE_LEVELS, means, stds):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold")

        ax.set_xlabel("Nivel de ruido (σ × std)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(max(0, np.nanmin(means) - 0.15), 1.08)
        ax.set_xticks(NOISE_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#fafbfc")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"Robustez bajo Ruido Gaussiano — {model_name}\n"
        f"Accuracy limpia: {mean_accs[0]:.3f} → con máximo ruido (σ=0.5): {mean_accs[-1]:.3f}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("+", "").replace("/", "_")
    path = os.path.join(output_dir, f"robustness_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica guardada: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación de robustez bajo ruido gaussiano"
    )
    parser.add_argument("--model", choices=["pose", "cnn", "all"],
                        default="pose",
                        help="Modelo a evaluar (default: pose)")
    parser.add_argument("--k", type=int, default=5,
                        help="Número de folds (default: 5)")
    parser.add_argument("--dataset", default=DATASET_AUG_DIR)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "robustness"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.model in ("pose", "all"):
        noise_robustness_pose(args.dataset, args.output, k=args.k)

    if args.model in ("cnn", "all"):
        noise_robustness_cnn(args.dataset, args.output, k=args.k)
