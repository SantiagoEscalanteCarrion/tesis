"""
Antecedente [4] — Duan et al. (2025) Dual AttentionUNet
=========================================================
Referencia:
    Duan X, et al. "Deep learning-assisted screening and diagnosis of
    scoliosis: segmentation of bare-back images via an attention-enhanced
    convolutional neural network."
    Journal of Orthopaedic Surgery and Research. 2025;20:PMC11827350.

Arquitectura reproducida:
    Dual AttentionUNet — encoder U-Net con:
      - Channel Attention: Squeeze-Excitation (SE) blocks
      - Spatial Self-Attention: mapa de atención convolucional
    Encoder: 4 etapas con filtros [64, 128, 256, 512], 2×Conv3×3 por etapa.
    Input: 256×256×3 (resolución estándar U-Net para imagen médica)

Adaptaciones documentadas:
    1. Tarea: segmentación + clasificación de severidad → binaria (scoliosis_yes/no)
       El dataset de Escalante no tiene máscaras de segmentación ni ángulo de Cobb.
    2. Se usa únicamente el encoder + dual attention + GAP + cabeza de clasificación.
       El decoder U-Net se omite al carecer de anotaciones de segmentación.
    3. Sin backbone preentrenado (U-Net original entrena desde cero para imagen médica).
    4. Normalización: división entre 255 (sin preprocess_input de ImageNet).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = (256, 256)


# ─────────────────────────────────────────────────────────────
# BLOQUES DE ATENCIÓN
# ─────────────────────────────────────────────────────────────

def _squeeze_excitation(x, ratio=16, name="se"):
    """
    Channel Attention — Squeeze-Excitation block (Hu et al., CVPR 2018).
    Pondera cada canal de features según su importancia global.
    """
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(se)
    se = layers.Dense(
        max(filters // ratio, 1), activation="relu",
        use_bias=False, name=f"{name}_fc1"
    )(se)
    se = layers.Dense(
        filters, activation="sigmoid",
        use_bias=False, name=f"{name}_fc2"
    )(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def _spatial_self_attention(x, name="spa"):
    """
    Spatial Self-Attention — mapa espacial convolucional (1×1 → sigmoid).
    Recalibra la importancia de cada posición espacial en el mapa de features.
    """
    att = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                        name=f"{name}_map")(x)
    return layers.Multiply(name=f"{name}_scale")([x, att])


# ─────────────────────────────────────────────────────────────
# BLOQUE ENCODER CON DUAL ATTENTION
# ─────────────────────────────────────────────────────────────

def _dual_attention_block(x, filters, name):
    """
    Bloque encoder del Dual AttentionUNet:
      Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU
      → Channel Attention (SE) → Spatial Self-Attention
    """
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same",
                      name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)

    x = _squeeze_excitation(x, ratio=16, name=f"{name}_se")
    x = _spatial_self_attention(x, name=f"{name}_spa")
    return x


# ─────────────────────────────────────────────────────────────
# MODELO COMPLETO
# ─────────────────────────────────────────────────────────────

def build_dual_attention_unet_classifier(dropout=0.5):
    """
    Encoder del Dual AttentionUNet adaptado para clasificación binaria.
    Arquitectura: 4×DualAttentionBlock + MaxPool → GAP → Dense(256) → Sigmoid

    Args:
        dropout: tasa de dropout en la cabeza de clasificación.
    Returns:
        tf.keras.Model compilable directamente.
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")

    # Normalización [0, 255] → [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Encoder — 4 etapas con doble atención
    x = _dual_attention_block(x, 64,  name="enc1")
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = _dual_attention_block(x, 128, name="enc2")
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = _dual_attention_block(x, 256, name="enc3")
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = _dual_attention_block(x, 512, name="enc4")

    # Cabeza de clasificación
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2, name="dropout_fc")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inputs, outputs=output, name="DualAttentionUNet_Classifier")
