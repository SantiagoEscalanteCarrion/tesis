"""
Antecedente [3] — Zhang et al. (2023) ScolioNets
==================================================
Referencia:
    Zhang T, Zhu C, et al. "Deep Learning Model to Classify and Monitor
    Idiopathic Scoliosis in Adolescents Using a Single Smartphone Photograph."
    JAMA Network Open. 2023;6(8):e2330617.

Arquitectura reproducida:
    ResNet50 + Attention Branch Network (ABN)
    Referencia de atención: Fukui et al. CVPR 2019 (ref [29] del paper)
    Input: 224×224×3 (especificado en el paper)

Adaptaciones necesarias por diferencia de dataset:
    1. Tarea: 3 clases de severidad → binaria (scoliosis_yes/no)
       Justificación: el dataset de Escalante no tiene ángulo de Cobb medido.
    2. Preprocesamiento: video matting omitido (modelo propietario no publicado).
    3. Backbone explícito: ResNet50 (ABN original, compatible con input 224×224).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


IMG_SIZE = (224, 224)


def _attention_branch(x, num_classes=1):
    """
    Rama de atención según Attention Branch Network (Fukui et al., CVPR 2019).
    Genera un mapa de atención espacial que pondera las features del backbone.

    Args:
        x: feature map del backbone, shape (B, H, W, C)
        num_classes: 1 para clasificación binaria

    Returns:
        attention_map: mapa de atención normalizado (B, H, W, 1)
        class_output:  predicción auxiliar desde la rama de atención
    """
    # Convolución 1×1 para reducir canales
    att = layers.Conv2D(512, (1, 1), padding="same", activation="relu",
                        name="att_conv1")(x)
    att = layers.BatchNormalization(name="att_bn1")(att)
    att = layers.Conv2D(256, (1, 1), padding="same", activation="relu",
                        name="att_conv2")(att)
    att = layers.BatchNormalization(name="att_bn2")(att)

    # Mapa de atención espacial: sigmoid normaliza a [0, 1]
    att_map = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                            name="att_map")(att)

    # Predicción auxiliar desde la rama de atención (multi-task)
    att_gap = layers.GlobalAveragePooling2D(name="att_gap")(att)
    att_out = layers.Dense(num_classes, activation="sigmoid",
                           name="att_output")(att_gap)

    return att_map, att_out


def build_scolionets(trainable_base=False, dropout=0.5):
    """
    Construye el modelo ScolioNets adaptado para clasificación binaria.

    Arquitectura:
        ResNet50 (backbone) → features (7×7×2048)
        ↓
        Attention Branch → mapa atención (7×7×1) + predicción auxiliar
        ↓
        features × attention_map  (ponderación espacial)
        ↓
        GlobalAveragePooling → Dense(256) → Dropout → Dense(1, sigmoid)

    Args:
        trainable_base: Si True, el backbone ResNet50 acepta gradientes.
        dropout: Tasa de dropout en la cabeza de clasificación.

    Returns:
        model Keras con entrada imagen (224×224×3) y dos salidas:
            - output_main: predicción principal (clasificación ponderada por atención)
            - att_output:  predicción auxiliar de la rama de atención
    """
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")

    # Extraer features del backbone (salida: 7×7×2048)
    features = backbone(inputs, training=trainable_base)

    # Attention Branch Network
    att_map, att_out = _attention_branch(features, num_classes=1)

    # Ponderar features con el mapa de atención
    attended = layers.Multiply(name="attended_features")([features, att_map])

    # Cabeza de clasificación principal
    x = layers.GlobalAveragePooling2D(name="gap")(attended)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2, name="dropout_fc")(x)
    main_out = layers.Dense(1, activation="sigmoid", name="output_main")(x)

    model = Model(
        inputs=inputs,
        outputs=[main_out, att_out],
        name="ScolioNets_Binary"
    )
    return model


def build_scolionets_single_output(trainable_base=False, dropout=0.5):
    """
    Versión simplificada con una sola salida (para evaluación con K-Fold CV).
    Misma arquitectura pero sin la salida auxiliar de la rama de atención.
    """
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    features = backbone(inputs, training=trainable_base)

    att_map, _ = _attention_branch(features, num_classes=1)
    attended = layers.Multiply(name="attended_features")([features, att_map])

    x = layers.GlobalAveragePooling2D(name="gap")(attended)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2, name="dropout_fc")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=output, name="ScolioNets_Binary_Single")
    return model
