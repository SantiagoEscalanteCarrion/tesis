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

Adaptaciones documentadas:
    1. Tarea: 3 clases de severidad → binaria (scoliosis_yes/no)
       El dataset de Escalante no tiene ángulo de Cobb medido.
    2. Preprocesamiento: video matting omitido (modelo propietario no publicado).
    3. Backbone: ResNet50 (eAppendix 4 no publicado; ABN original usa ResNet).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

IMG_SIZE = (224, 224)


def _attention_branch(x, num_classes=1):
    """
    Rama de atención — Attention Branch Network (Fukui et al., CVPR 2019).
    Genera mapa de atención espacial que pondera las features del backbone.
    """
    att = layers.Conv2D(512, (1, 1), padding="same", activation="relu",
                        name="att_conv1")(x)
    att = layers.BatchNormalization(name="att_bn1")(att)
    att = layers.Conv2D(256, (1, 1), padding="same", activation="relu",
                        name="att_conv2")(att)
    att = layers.BatchNormalization(name="att_bn2")(att)
    att_map = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                            name="att_map")(att)
    att_gap = layers.GlobalAveragePooling2D(name="att_gap")(att)
    att_out = layers.Dense(num_classes, activation="sigmoid",
                           name="att_output")(att_gap)
    return att_map, att_out


def build_scolionets_single_output(trainable_base=False, dropout=0.5):
    """
    ScolioNets adaptado para clasificación binaria (una sola salida).
    Arquitectura: ResNet50 → ABN attention → GAP → Dense(256) → Sigmoid
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

    return Model(inputs=inputs, outputs=output, name="ScolioNets_Binary")
