"""
Enfoque 3 — Modelo Híbrido: CNN + Pose Features
================================================
Arquitectura de dos streams paralelos:

  Stream A — Visual (CNN):
      Imagen (224×224×3) → EfficientNetB0 (frozen/fine-tuned)
      → GlobalAveragePooling → Dense(256) → Dropout → Dense(128)

  Stream B — Clínico (Pose):
      Vector de features geométricas (12,) → Dense(64) → Dense(32)

  Fusión:
      Concat([A, B]) → Dense(64) → Dropout → Dense(32) → Sigmoid

El ablation study compara:
  CNN solo (E1) vs Pose solo (E2) vs Híbrido (E3)

Uso:
    from model_hybrid import HybridModel
    model = HybridModel()
    model.build()
    model.train(train_data, val_data)
    model.evaluate(test_data)
    model.ablation_study(results_e1, results_e2)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS_HEAD, EPOCHS_FINE,
    LEARNING_RATE_HEAD, LEARNING_RATE_FINE,
    NUM_POSE_FEATURES, OUTPUT_DIR, SEED, TEST_SPLIT, VAL_SPLIT
)


# ─────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS DUAL
# ─────────────────────────────────────────────────────────────

def build_hybrid_dataset(dataset_dir, pose_features_path,
                          val_split=VAL_SPLIT, test_split=TEST_SPLIT,
                          seed=SEED):
    """
    Construye datasets duales (imagen + features de pose) alineados.

    Args:
        dataset_dir:        Carpeta raíz del dataset con subcarpetas por clase.
        pose_features_path: Archivo .pkl generado por model_pose.py con
                            {X, y, paths}.

    Returns:
        (train_ds, val_ds, test_ds) — cada elemento es un tf.data.Dataset
        que emite ((imagen_tensor, pose_tensor), label)
    """
    # Cargar features de pose pre-calculadas
    with open(pose_features_path, "rb") as f:
        pose_data = pickle.load(f)
    pose_X     = pose_data["X"].astype(np.float32)     # (N, 12)
    pose_y     = pose_data["y"].astype(np.float32)     # (N,)
    pose_paths = pose_data["paths"]                    # lista de str

    # Normalizar features de pose
    mean = pose_X.mean(axis=0)
    std  = pose_X.std(axis=0) + 1e-8
    pose_X_norm = (pose_X - mean) / std

    # Guardar estadísticas para inferencia futura
    norm_stats = {"mean": mean, "std": std}

    # Split estratificado
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(pose_y))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_split, stratify=pose_y, random_state=seed
    )
    val_fraction = val_split / (1 - test_split)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_fraction,
        stratify=pose_y[idx_trainval], random_state=seed
    )

    def make_ds(idx, shuffle=False):
        """Crea tf.data.Dataset para un subconjunto de índices."""
        img_paths = [pose_paths[i] for i in idx]
        feats     = pose_X_norm[idx]
        labels    = pose_y[idx]

        def load_and_preprocess(path, feat, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            return (img, feat), label

        ds = tf.data.Dataset.from_tensor_slices((
            img_paths, feats, labels
        ))
        ds = ds.map(
            lambda p, f, l: load_and_preprocess(p, f, l),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if shuffle:
            ds = ds.shuffle(1000, seed=seed)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(idx_train, shuffle=True)
    val_ds   = make_ds(idx_val)
    test_ds  = make_ds(idx_test)

    print(f"Hybrid dataset — Train: {len(idx_train)} | "
          f"Val: {len(idx_val)} | Test: {len(idx_test)}")

    return train_ds, val_ds, test_ds, norm_stats


# ─────────────────────────────────────────────────────────────
# ARQUITECTURA DEL MODELO HÍBRIDO
# ─────────────────────────────────────────────────────────────

def build_hybrid_model(trainable_base=False, dropout=0.4,
                        pose_feature_dim=NUM_POSE_FEATURES):
    """
    Construye el modelo híbrido de dos streams.

    Args:
        trainable_base:    Si True, el backbone CNN acepta gradientes.
        dropout:           Tasa de dropout en la cabeza de fusión.
        pose_feature_dim:  Dimensión del vector de features de pose.

    Returns:
        modelo Keras de dos entradas (imagen, pose_features)
    """
    # ── Stream A: Visual (CNN) ────────────────────────────────
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = trainable_base

    img_input = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")

    x_img = backbone(img_input, training=trainable_base)
    x_img = layers.GlobalAveragePooling2D(name="gap")(x_img)
    x_img = layers.BatchNormalization(name="bn_cnn")(x_img)
    x_img = layers.Dense(256, activation="relu", name="cnn_fc1")(x_img)
    x_img = layers.Dropout(dropout, name="dropout_cnn")(x_img)
    x_img = layers.Dense(128, activation="relu", name="cnn_fc2")(x_img)

    # ── Stream B: Clínico (Pose Features) ────────────────────
    pose_input = tf.keras.Input(shape=(pose_feature_dim,), name="pose_input")

    x_pose = layers.Dense(64, activation="relu", name="pose_fc1")(pose_input)
    x_pose = layers.BatchNormalization(name="bn_pose")(x_pose)
    x_pose = layers.Dense(32, activation="relu", name="pose_fc2")(x_pose)

    # ── Fusión ────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([x_img, x_pose])
    fused = layers.Dense(64, activation="relu", name="fusion_fc1")(fused)
    fused = layers.Dropout(dropout / 2, name="dropout_fusion")(fused)
    fused = layers.Dense(32, activation="relu", name="fusion_fc2")(fused)
    output = layers.Dense(1, activation="sigmoid", name="output")(fused)

    model = Model(
        inputs=[img_input, pose_input],
        outputs=output,
        name="ScoliosisHybrid"
    )
    return model


# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO EN DOS FASES
# ─────────────────────────────────────────────────────────────

def get_callbacks_hybrid(phase_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f"hybrid_{phase_name}_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]


def train_hybrid(dataset_dir, pose_features_path, output_dir=None):
    """
    Entrena el modelo híbrido en dos fases (igual que model_cnn.py).

    Returns:
        model, history1, history2, test_ds
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "hybrid")
    os.makedirs(output_dir, exist_ok=True)

    train_ds, val_ds, test_ds, norm_stats = build_hybrid_dataset(
        dataset_dir, pose_features_path
    )

    # Guardar estadísticas de normalización para inferencia
    with open(os.path.join(output_dir, "pose_norm_stats.pkl"), "wb") as f:
        pickle.dump(norm_stats, f)

    # ── FASE 1: Solo cabeza ────────────────────────────────────
    print("\n" + "="*55)
    print("  FASE 1 — Head Training (backbone congelado)")
    print("="*55)
    model = build_hybrid_model(trainable_base=False)
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks_hybrid("head", output_dir),
    )

    # ── FASE 2: Fine-tuning ────────────────────────────────────
    print("\n" + "="*55)
    print("  FASE 2 — Fine-tuning (backbone parcialmente descongelado)")
    print("="*55)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers:
        if not layer.name.startswith(("block6", "block7", "top")):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE_FINE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks_hybrid("finetune", output_dir),
    )

    final_path = os.path.join(output_dir, "hybrid_final.keras")
    model.save(final_path)
    print(f"\nModelo guardado en: {final_path}")

    return model, history1, history2, test_ds


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────

def evaluate_hybrid(model, test_ds, output_dir=None, threshold=0.5):
    """Evalúa el modelo híbrido en el conjunto de test."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "hybrid")
    os.makedirs(output_dir, exist_ok=True)

    y_true, y_prob = [], []
    for (images, poses), labels in test_ds:
        probs = model.predict([images, poses], verbose=0)
        y_prob.extend(probs.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "="*55)
    print("  EVALUACIÓN — Modelo Híbrido")
    print("="*55)
    print(classification_report(y_true, y_pred,
                                target_names=["scoliosis_no", "scoliosis_yes"]))
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    _plot_confusion_matrix(y_true, y_pred, output_dir, "Hybrid")
    _plot_roc_curve(y_true, y_prob, auc, output_dir, "Hybrid")

    return {"accuracy": np.mean(y_true == y_pred), "auc": auc,
            "y_true": y_true, "y_prob": y_prob}


# ─────────────────────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────────────────────

def ablation_study(results_cnn, results_pose, results_hybrid, output_dir=None):
    """
    Compara los 3 enfoques en una sola gráfica.

    Args:
        results_*: dicts retornados por evaluate_*() con claves
                   'accuracy', 'auc', 'y_true', 'y_prob'
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "E1: CNN": results_cnn,
        "E2: Pose+ML": results_pose,
        "E3: Híbrido": results_hybrid,
    }

    # ── Tabla resumen ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  ABLATION STUDY — Comparativa de los 3 enfoques")
    print("="*55)
    print(f"{'Modelo':<15} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("─"*40)
    for name, res in models.items():
        print(f"{name:<15} {res['accuracy']:>10.4f} {res['auc']:>10.4f}")

    # ── Gráfica de barras comparativa ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(models.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    accs = [models[n]["accuracy"] for n in names]
    aucs = [models[n]["auc"] for n in names]

    for ax, values, title, ylabel in [
        (axes[0], accs, "Accuracy en Test", "Accuracy"),
        (axes[1], aucs, "AUC-ROC en Test",  "AUC-ROC"),
    ]:
        bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Ablation Study — Detección de Escoliosis", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_comparison.png"), dpi=150)
    plt.show()

    # ── Curvas ROC superpuestas ────────────────────────────────
    plt.figure(figsize=(8, 6))
    for (name, res), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        plt.plot(fpr, tpr, lw=2, color=color,
                 label=f"{name} (AUC={res['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC — Ablation Study")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_roc_curves.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# GRÁFICAS AUXILIARES
# ─────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, output_dir, model_name=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["no", "yes"],
                yticklabels=["no", "yes"], ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                f"{model_name.lower()}_confusion.png"), dpi=150)
    plt.show()


def _plot_roc_curve(y_true, y_prob, auc, output_dir, model_name=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, color="#55A868", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"Curva ROC — {model_name}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                f"{model_name.lower()}_roc.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA DIRECTO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DATASET_AUG_DIR

    out = os.path.join(OUTPUT_DIR, "hybrid")
    features_path = os.path.join(OUTPUT_DIR, "pose", "pose_features.pkl")

    if not os.path.exists(features_path):
        print("ADVERTENCIA: Primero ejecuta model_pose.py para generar las features.")
        print("  python model_pose.py")
        exit(1)

    model, h1, h2, test_ds = train_hybrid(DATASET_AUG_DIR, features_path, out)
    results = evaluate_hybrid(model, test_ds, out)
    print(f"\nAccuracy en test: {results['accuracy']:.4f}")
    print(f"AUC-ROC en test:  {results['auc']:.4f}")
