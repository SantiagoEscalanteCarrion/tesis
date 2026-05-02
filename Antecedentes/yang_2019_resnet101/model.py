"""
Antecedente [5] — Yang et al. (2019) ResNet-101 para escoliosis
================================================================
Referencia:
    Yang J, Zhang K, et al. "Development and validation of deep learning
    algorithms for scoliosis screening using back images."
    Communications Biology. 2019;2:390.
    DOI: 10.1038/s42003-019-0635-8

Arquitectura reproducida:
    Pipeline original: Faster R-CNN (localización) → ResNet-101 (clasificación)
    Backbone elegido tras comparar AlexNet, VGG16, Inception-V4 y ResNet-101.
    Framework original: Caffe con pesos ImageNet (fine-tuning).
    Input: 224×224×3 (estándar ResNet).

Adaptaciones documentadas:
    1. Faster R-CNN omitido: el dataset de Escalante ya contiene imágenes
       recortadas de la espalda, por lo que la detección de región no es necesaria.
    2. Tarea: clasificación de severidad (4 clases) → binaria (scoliosis_yes/no).
       El dataset de Escalante no tiene ángulo de Cobb medido.
    3. Framework: TensorFlow/Keras en lugar de Caffe.
    4. Fine-tuning: bloque conv5 de ResNet-101 (mismo criterio que Zhang et al. 2023).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet101

IMG_SIZE = (224, 224)


def build_yang2019_resnet101(trainable_base=False, dropout=0.5):
    """
    ResNet-101 fine-tuned para clasificación binaria de escoliosis.
    Arquitectura: ResNet101 (ImageNet) → GAP → Dense(256) → Sigmoid

    Args:
        trainable_base: si True, el backbone es entrenable (fase fine-tuning).
        dropout: tasa de dropout en la cabeza de clasificación.
    Returns:
        tf.keras.Model listo para compilar.
    """
    backbone = ResNet101(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = backbone(inputs, training=trainable_base)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2, name="dropout_fc")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inputs, outputs=output, name="Yang2019_ResNet101_Binary")
