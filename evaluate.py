"""
Script de evaluación independiente
=====================================
Carga modelos ya entrenados y genera métricas e informes completos.
Útil para evaluar con nuevas imágenes o para regenerar gráficas.

Uso:
    # Evaluar CNN
    python evaluate.py --model cnn --model-path outputs/cnn/cnn_final.keras

    # Evaluar Pose
    python evaluate.py --model pose --model-path outputs/pose/pose_RandomForest.pkl

    # Evaluar Híbrido
    python evaluate.py --model hybrid --model-path outputs/hybrid/hybrid_final.keras

    # Inferencia sobre una sola imagen
    python evaluate.py --predict --image ruta/imagen.jpg --model cnn \
                       --model-path outputs/cnn/cnn_final.keras
"""

import argparse
import os
import pickle
import numpy as np
import cv2
import tensorflow as tf

from config import (
    IMG_SIZE, DATASET_AUG_DIR, OUTPUT_DIR,
    POSE_FEATURE_NAMES, CLASS_LABELS
)


# ─────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────

def load_cnn(model_path):
    print(f"Cargando CNN desde: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_pose_model(model_path):
    print(f"Cargando modelo Pose desde: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_hybrid(model_path):
    print(f"Cargando Híbrido desde: {model_path}")
    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────────────────────
# INFERENCIA SOBRE UNA IMAGEN
# ─────────────────────────────────────────────────────────────

def predict_single_image_cnn(model, image_path, threshold=0.5):
    """
    Predice la clase de una imagen con el modelo CNN.
    Retorna: (label, confidence)
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array, verbose=0)[0, 0])
    label = CLASS_LABELS[int(prob >= threshold)]
    return label, prob


def predict_single_image_pose(pipeline, image_path, norm_stats=None, threshold=0.5):
    """
    Predice la clase de una imagen con el clasificador de pose.
    Retorna: (label, confidence)
    """
    import mediapipe as mp
    from model_pose import extract_pose_features_from_image

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2,
        min_detection_confidence=0.5
    )
    feats = extract_pose_features_from_image(image_path, mp_pose)
    mp_pose.close()

    if feats is None:
        return "ERROR: no se detectó pose en la imagen", 0.0

    if norm_stats:
        feats = (feats - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)

    feats = feats.reshape(1, -1)
    prob = float(pipeline.predict_proba(feats)[0, 1])
    label = CLASS_LABELS[int(prob >= threshold)]
    return label, prob


def predict_single_image_hybrid(model, image_path, pose_pipeline,
                                 norm_stats, threshold=0.5):
    """
    Predice la clase de una imagen con el modelo híbrido.
    Requiere tanto el modelo Keras como el pipeline de pose (para features).
    Retorna: (label, confidence)
    """
    import mediapipe as mp
    from model_pose import extract_pose_features_from_image

    # Imagen
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Pose features
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2,
        min_detection_confidence=0.5
    )
    feats = extract_pose_features_from_image(image_path, mp_pose)
    mp_pose.close()

    if feats is None:
        return "ERROR: no se detectó pose en la imagen", 0.0

    feats = (feats - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
    feats = feats.reshape(1, -1).astype(np.float32)

    prob = float(model.predict([img_array, feats], verbose=0)[0, 0])
    label = CLASS_LABELS[int(prob >= threshold)]
    return label, prob


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN COMPLETA SOBRE DATASET
# ─────────────────────────────────────────────────────────────

def evaluate_cnn_full(model_path, dataset_dir, output_dir):
    from model_cnn import build_datasets, evaluate_cnn
    model = load_cnn(model_path)
    _, _, test_ds = build_datasets(dataset_dir)
    return evaluate_cnn(model, test_ds, output_dir)


def evaluate_pose_full(model_path, features_path, output_dir):
    from model_pose import split_data, evaluate_pose_classifiers

    pipeline = load_pose_model(model_path)
    model_name = os.path.basename(model_path).replace("pose_", "").replace(".pkl", "")

    with open(features_path, "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    _, _, X_test, _, _, y_test = split_data(X, y)

    return evaluate_pose_classifiers({model_name: pipeline}, X_test, y_test, output_dir)


def evaluate_hybrid_full(model_path, dataset_dir, features_path,
                          norm_stats_path, output_dir):
    from model_hybrid import build_hybrid_dataset, evaluate_hybrid

    model = load_hybrid(model_path)
    with open(norm_stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    _, _, test_ds, _ = build_hybrid_dataset(dataset_dir, features_path)
    return evaluate_hybrid(model, test_ds, output_dir)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos de detección de escoliosis"
    )
    parser.add_argument("--model", choices=["cnn", "pose", "hybrid"], required=True)
    parser.add_argument("--model-path", required=True,
                        help="Ruta al archivo del modelo (.keras o .pkl)")
    parser.add_argument("--dataset", default=DATASET_AUG_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)

    # Para predicción de imagen única
    parser.add_argument("--predict", action="store_true",
                        help="Modo predicción: evalúa una sola imagen")
    parser.add_argument("--image", default=None,
                        help="Ruta a la imagen (solo con --predict)")
    parser.add_argument("--pose-features",
                        default=None,
                        help="Ruta a pose_features.pkl (para hybrid/pose eval)")
    parser.add_argument("--norm-stats",
                        default=None,
                        help="Ruta a pose_norm_stats.pkl (para hybrid predict)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.image:
            print("ERROR: Debes indicar --image con --predict")
            exit(1)

        if args.model == "cnn":
            model = load_cnn(args.model_path)
            label, prob = predict_single_image_cnn(model, args.image)
            print(f"\nPredicción: {label}")
            print(f"Confianza:  {prob:.4f} ({'scoliosis_yes' if prob >= 0.5 else 'scoliosis_no'})")

        elif args.model == "pose":
            pipeline = load_pose_model(args.model_path)
            norm_stats = None
            if args.norm_stats:
                with open(args.norm_stats, "rb") as f:
                    norm_stats = pickle.load(f)
            label, prob = predict_single_image_pose(pipeline, args.image, norm_stats)
            print(f"\nPredicción: {label}")
            print(f"Confianza:  {prob:.4f}")

        elif args.model == "hybrid":
            if not args.norm_stats:
                print("ERROR: Para el modelo híbrido necesitas --norm-stats")
                exit(1)
            model = load_hybrid(args.model_path)
            with open(args.norm_stats, "rb") as f:
                norm_stats = pickle.load(f)
            label, prob = predict_single_image_hybrid(
                model, args.image, None, norm_stats
            )
            print(f"\nPredicción: {label}")
            print(f"Confianza:  {prob:.4f}")

    else:
        out = os.path.join(args.output, args.model)
        os.makedirs(out, exist_ok=True)

        if args.model == "cnn":
            evaluate_cnn_full(args.model_path, args.dataset, out)

        elif args.model == "pose":
            feats = args.pose_features or os.path.join(args.output, "pose", "pose_features.pkl")
            evaluate_pose_full(args.model_path, feats, out)

        elif args.model == "hybrid":
            feats = args.pose_features or os.path.join(args.output, "pose", "pose_features.pkl")
            norm  = args.norm_stats     or os.path.join(args.output, "hybrid", "pose_norm_stats.pkl")
            evaluate_hybrid_full(args.model_path, args.dataset, feats, norm, out)
