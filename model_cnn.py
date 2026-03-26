"""
Enfoque 1 — Transfer Learning CNN
===================================
Usa EfficientNetB0 preentrenado en ImageNet y lo fine-tunea para
clasificación binaria de escoliosis (scoliosis_yes / scoliosis_no).

Flujo de entrenamiento en dos fases:
  Fase 1 — Head training:  base congelada, solo se entrena la cabeza
  Fase 2 — Fine-tuning:    se descongela el último bloque del backbone

Uso:
    from model_cnn import ScoliosisCNN
    model = ScoliosisCNN()
    model.build()
    model.train(train_ds, val_ds)
    model.evaluate(test_ds)
    model.grad_cam(image_path)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns

from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS_HEAD, EPOCHS_FINE,
    LEARNING_RATE_HEAD, LEARNING_RATE_FINE, OUTPUT_DIR, SEED
)


# ─────────────────────────────────────────────────────────────
# UTILIDADES DE DATASET
# ─────────────────────────────────────────────────────────────

def build_datasets(dataset_dir, val_split=0.15, test_split=0.15, seed=SEED):
    """
    Carga imágenes desde dataset_dir con estructura:
        dataset_dir/
            scoliosis_no/
            scoliosis_yes/

    Retorna: (train_ds, val_ds, test_ds)
    """
    # Separamos test primero para que nunca "contamine" entrenamiento
    full_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=IMG_SIZE,
        batch_size=None,          # sin batch para poder hacer split manual
        shuffle=True,
        seed=seed,
        label_mode="binary",
    )

    total = tf.data.experimental.cardinality(full_ds).numpy()
    test_size = int(total * test_split)
    val_size  = int(total * val_split)
    train_size = total - test_size - val_size

    train_ds = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds   = remaining.take(val_size)
    test_ds  = remaining.skip(val_size)

    # Preprocesamiento y batching
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000, seed=seed)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

    val_ds = (val_ds
              .map(preprocess, num_parallel_calls=AUTOTUNE)
              .cache()
              .batch(BATCH_SIZE)
              .prefetch(AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────
# DEFINICIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────

def build_cnn_model(trainable_base=False, dropout=0.4):
    """
    Construye el modelo CNN sobre EfficientNetB0.

    Args:
        trainable_base: Si True, el backbone acepta gradientes.
        dropout: Tasa de dropout antes de la capa de clasificación.

    Returns:
        modelo Keras compilado
    """
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")

    # Augmentation ligera solo durante entrenamiento
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    x = backbone(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="ScoliosisCNN")
    return model


# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO EN DOS FASES
# ─────────────────────────────────────────────────────────────

def get_callbacks(phase_name, output_dir):
    """Retorna lista de callbacks para cada fase de entrenamiento."""
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
            filepath=os.path.join(output_dir, f"cnn_{phase_name}_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]


def train_cnn(dataset_dir, output_dir=None):
    """
    Entrena el modelo CNN en dos fases y guarda el modelo final.

    Args:
        dataset_dir: Carpeta raíz del dataset augmentado.
        output_dir:  Carpeta donde se guardan modelos y gráficas.

    Returns:
        model, history_phase1, history_phase2, test_ds
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "cnn")
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos
    print("Cargando dataset...")
    train_ds, val_ds, test_ds = build_datasets(dataset_dir)

    # ── FASE 1: Solo se entrena la cabeza ──────────────────────
    print("\n" + "="*55)
    print("  FASE 1 — Head Training (backbone congelado)")
    print("="*55)
    model = build_cnn_model(trainable_base=False)
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
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks("head", output_dir),
    )

    # ── FASE 2: Fine-tuning de las últimas capas ───────────────
    print("\n" + "="*55)
    print("  FASE 2 — Fine-tuning (backbone parcialmente descongelado)")
    print("="*55)

    # Descongelar solo el bloque 6 y 7 de EfficientNetB0
    backbone = model.layers[3]   # índice del backbone en el modelo
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
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks("finetune", output_dir),
    )

    # Guardar modelo final
    final_path = os.path.join(output_dir, "cnn_final.keras")
    model.save(final_path)
    print(f"\nModelo guardado en: {final_path}")

    # Graficar curvas de aprendizaje
    _plot_training_curves(history1, history2, output_dir)

    return model, history1, history2, test_ds


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────

def evaluate_cnn(model, test_ds, output_dir=None, threshold=0.5):
    """
    Evalúa el modelo CNN en el conjunto de test.
    Imprime y guarda: accuracy, F1, AUC-ROC, matriz de confusión.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "cnn")
    os.makedirs(output_dir, exist_ok=True)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.extend(probs.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "="*55)
    print("  EVALUACIÓN — CNN")
    print("="*55)
    print(classification_report(y_true, y_pred,
                                target_names=["scoliosis_no", "scoliosis_yes"]))
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    _plot_confusion_matrix(y_true, y_pred, output_dir, model_name="CNN")
    _plot_roc_curve(y_true, y_prob, auc, output_dir, model_name="CNN")

    return {"accuracy": np.mean(y_true == y_pred), "auc": auc,
            "y_true": y_true, "y_prob": y_prob}


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

def compute_grad_cam(model, image_path, layer_name="top_activation",
                     output_dir=None):
    """
    Genera un mapa de calor Grad-CAM sobre una imagen de entrada.

    Args:
        model:      Modelo CNN entrenado.
        image_path: Ruta a la imagen JPG.
        layer_name: Nombre de la capa de convolución de referencia.
        output_dir: Carpeta donde guardar la imagen resultante.
    """
    # Preprocesar imagen
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Modelo auxiliar hasta la capa objetivo
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)

    # Superponer sobre imagen original
    orig_img = cv2.imread(image_path)
    orig_img = cv2.resize(orig_img, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

    pred_prob = float(predictions[0, 0])
    label = "scoliosis_yes" if pred_prob >= 0.5 else "scoliosis_no"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Mapa de calor Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Superposición\nPredicción: {label} ({pred_prob:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(image_path).replace(".jpg", "_gradcam.png")
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.show()
    return heatmap


# ─────────────────────────────────────────────────────────────
# GRÁFICAS AUXILIARES
# ─────────────────────────────────────────────────────────────

def _plot_training_curves(history1, history2, output_dir):
    acc1  = history1.history["accuracy"]
    acc2  = history2.history["accuracy"]
    vacc1 = history1.history["val_accuracy"]
    vacc2 = history2.history["val_accuracy"]
    loss1  = history1.history["loss"]
    loss2  = history2.history["loss"]
    vloss1 = history1.history["val_loss"]
    vloss2 = history2.history["val_loss"]

    epochs1 = range(1, len(acc1) + 1)
    epochs2 = range(len(acc1) + 1, len(acc1) + len(acc2) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs1, acc1,  "b-",  label="Train (Fase 1)")
    axes[0].plot(epochs1, vacc1, "b--", label="Val (Fase 1)")
    axes[0].plot(epochs2, acc2,  "r-",  label="Train (Fase 2 FT)")
    axes[0].plot(epochs2, vacc2, "r--", label="Val (Fase 2 FT)")
    axes[0].axvline(len(acc1), color="gray", linestyle=":", label="Inicio FT")
    axes[0].set_title("Accuracy durante entrenamiento")
    axes[0].set_xlabel("Épocas"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs1, loss1,  "b-",  label="Train (Fase 1)")
    axes[1].plot(epochs1, vloss1, "b--", label="Val (Fase 1)")
    axes[1].plot(epochs2, loss2,  "r-",  label="Train (Fase 2 FT)")
    axes[1].plot(epochs2, vloss2, "r--", label="Val (Fase 2 FT)")
    axes[1].axvline(len(acc1), color="gray", linestyle=":", label="Inicio FT")
    axes[1].set_title("Loss durante entrenamiento")
    axes[1].set_xlabel("Épocas"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("CNN — Curvas de Aprendizaje", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnn_training_curves.png"), dpi=150)
    plt.show()


def _plot_confusion_matrix(y_true, y_pred, output_dir, model_name=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "yes"],
                yticklabels=["no", "yes"], ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                f"{model_name.lower()}_confusion_matrix.png"), dpi=150)
    plt.show()


def _plot_roc_curve(y_true, y_prob, auc, output_dir, model_name=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC — {model_name}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                f"{model_name.lower()}_roc_curve.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA DIRECTO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DATASET_AUG_DIR

    output_dir = os.path.join(OUTPUT_DIR, "cnn")
    model, h1, h2, test_ds = train_cnn(DATASET_AUG_DIR, output_dir)
    results = evaluate_cnn(model, test_ds, output_dir)
    print(f"\nAccuracy en test: {results['accuracy']:.4f}")
    print(f"AUC-ROC en test:  {results['auc']:.4f}")
