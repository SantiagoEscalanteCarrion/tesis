"""
Configuración global compartida por todos los modelos.
Ajusta BASE_DIR según tu ruta local en Windows.
"""

import os

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

# Ruta raíz del proyecto en tu máquina Windows
BASE_DIR = r"C:/Users/SANTIAGO/Desktop/TesisCodigo"

# Dataset original (pre-augmentation)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Dataset augmentado (generado por data_augmentation.py)
DATASET_AUG_DIR = os.path.join(BASE_DIR, "dataset_augmented")

# Directorio donde se guardan modelos entrenados y resultados
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Clases (deben coincidir con los nombres de subcarpetas)
CLASSES = ["scoliosis_no", "scoliosis_yes"]   # 0 = no, 1 = yes
CLASS_LABELS = {0: "scoliosis_no", 1: "scoliosis_yes"}

# ─────────────────────────────────────────────
# IMÁGENES
# ─────────────────────────────────────────────

IMG_SIZE = (224, 224)      # Tamaño de entrada para CNN (EfficientNetB0)
IMG_CHANNELS = 3

# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

SEED       = 42
TEST_SPLIT = 0.15          # 15 % para test
VAL_SPLIT  = 0.15          # 15 % para validación (del total)
BATCH_SIZE = 32
EPOCHS_HEAD  = 10          # Epochs entrenando solo la cabeza (base frozen)
EPOCHS_FINE  = 20          # Epochs de fine-tuning (base descongelada parcialmente)
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

# ─────────────────────────────────────────────
# POSE ESTIMATION
# ─────────────────────────────────────────────

# Índices de landmarks MediaPipe que usamos
MP_LANDMARKS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
}

# Features geométricas extraídas (ver model_pose.py)
POSE_FEATURE_NAMES = [
    "shoulder_height_diff",      # |Ly_shoulder - Ry_shoulder|
    "hip_height_diff",           # |Ly_hip - Ry_hip|
    "elbow_height_diff",         # |Ly_elbow - Ry_elbow|
    "wrist_height_diff",         # |Ly_wrist - Ry_wrist|
    "shoulder_width",            # distancia horizontal entre hombros
    "hip_width",                 # distancia horizontal entre caderas
    "shoulder_asymmetry_ratio",  # shoulder_height_diff / shoulder_width
    "hip_asymmetry_ratio",       # hip_height_diff / hip_width
    "trunk_tilt_x",              # desplazamiento x: midHip → midShoulder
    "trunk_tilt_angle",          # ángulo de inclinación del tronco (grados)
    "spine_deviation",           # desplazamiento lateral de la línea media
    "body_height_proxy",         # distancia vertical total (shoulder → hip)
]

NUM_POSE_FEATURES = len(POSE_FEATURE_NAMES)
