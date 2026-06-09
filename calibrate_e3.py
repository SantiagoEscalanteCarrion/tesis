"""
Calibración de E3 mediante Temperature Scaling.

Estima el parámetro T que minimiza la NLL en el validation set y lo guarda en
outputs/hybrid/e3_temperature.pkl. Corre este script UNA VEZ antes de arrancar
la webapp si quieres E3 calibrado.

Uso:
    python calibrate_e3.py
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR, DATASET_AUG_DIR


def _prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(_sigmoid(logits / T), eps, 1 - eps)
    return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def calibrate(model_path, features_path, norm_stats_path, output_path):
    import tensorflow as tf
    from model_hybrid import build_hybrid_dataset

    print("Cargando modelo E3...")
    model = tf.keras.models.load_model(model_path)

    print("Construyendo validation set...")
    _, val_ds, _, _ = build_hybrid_dataset(DATASET_AUG_DIR, features_path)

    print("Recogiendo probabilidades en validation set...")
    probs_list, labels_list = [], []
    for (img_batch, pose_batch), label_batch in val_ds:
        p = model.predict([img_batch, pose_batch], verbose=0)[:, 0]
        probs_list.append(p)
        labels_list.append(label_batch.numpy())

    probs  = np.concatenate(probs_list).astype(np.float64)
    labels = np.concatenate(labels_list).astype(np.float64)
    logits = _prob_to_logit(probs)

    print(f"Ejemplos de validación: {len(probs)}  |  NLL sin calibrar: {_nll(1.0, logits, labels):.4f}")

    result = minimize_scalar(
        lambda T: _nll(T, logits, labels),
        bounds=(1.0, 20.0),   # T < 1 amplifica logits → peor para out-of-distribution
        method="bounded",
    )
    T_opt = float(result.x)
    print(f"T óptimo: {T_opt:.4f}  |  NLL calibrado: {_nll(T_opt, logits, labels):.4f}")

    if abs(T_opt - 1.0) < 0.01:
        print("AVISO: T ≈ 1.0 — el modelo ya está bien calibrado en el validation set.")
        print("       El overconfidence observado es domain shift (fotos out-of-distribution).")
        print("       Temperature Scaling no lo puede corregir desde el validation set.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"T": T_opt}, f)
    print(f"Guardado en: {output_path}")
    return T_opt


if __name__ == "__main__":
    calibrate(
        model_path     = os.path.join(OUTPUT_DIR, "hybrid", "hybrid_final.keras"),
        features_path  = os.path.join(OUTPUT_DIR, "pose", "split_features.pkl"),
        norm_stats_path= os.path.join(OUTPUT_DIR, "hybrid", "pose_norm_stats.pkl"),
        output_path    = os.path.join(OUTPUT_DIR, "hybrid", "e3_temperature.pkl"),
    )
