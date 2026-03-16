"""
Data Augmentation para dataset de detección de escoliosis.
Genera imágenes aumentadas de forma conservadora para no alterar
características anatómicas relevantes.

Estructura esperada:
    TesisCodigo/
    └── dataset/
        ├── scoliosis_yes/   (120 imágenes JPG)
        └── scoliosis_no/    (203 imágenes JPG)

Salida:
    TesisCodigo/
    └── dataset_augmented/
        ├── scoliosis_yes/   (~1000 imágenes JPG)
        └── scoliosis_no/    (~1000 imágenes JPG)

Dependencias:
    pip install albumentations pillow numpy tqdm
"""

import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BASE_DIR    = r"C:/Users/SANTIAGO/Desktop/TesisCodigo"
INPUT_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "dataset_augmented")
TARGET      = 1000   # imágenes totales por clase (originales + augmentadas)
SEED        = 42
IMG_EXT     = ".jpg"

CLASSES = ["scoliosis_yes", "scoliosis_no"]

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# PIPELINE DE AUGMENTATION
# Parámetros conservadores: no se distorsiona
# la curvatura de la columna ni la postura.
# ─────────────────────────────────────────────

augment = A.Compose([
    # Flip horizontal: válido, la escoliosis puede curvarse a ambos lados
    A.HorizontalFlip(p=0.5),

    # Rotación leve: simula ligera inclinación del paciente al fotografiarse
    A.Rotate(limit=12, border_mode=0, p=0.7),

    # Traslación y zoom leve: simula variación de distancia/posición de cámara
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.08,
        rotate_limit=0,      # la rotación ya está arriba
        border_mode=0,
        p=0.6
    ),

    # Brillo y contraste leve: variación de iluminación entre tomas
    A.RandomBrightnessContrast(
        brightness_limit=0.12,
        contrast_limit=0.12,
        p=0.5
    ),

    # Ruido gaussiano muy suave: simula variaciones del sensor de la cámara
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),

    # Blur muy suave ocasional: simula leve desenfoque de cámara
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
])


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def load_image_paths(folder):
    """Retorna lista de rutas de imágenes JPG en la carpeta."""
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXT)
    ]
    return sorted(paths)


def save_image(img_array, path):
    """Guarda un numpy array como imagen JPG."""
    Image.fromarray(img_array).save(path, format="JPEG", quality=95)


def augment_class(class_name):
    """
    Copia los originales y genera imágenes augmentadas hasta alcanzar TARGET.
    """
    src_folder = os.path.join(INPUT_DIR, class_name)
    dst_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    image_paths = load_image_paths(src_folder)
    n_originals = len(image_paths)

    if n_originals == 0:
        print(f"[ADVERTENCIA] No se encontraron imágenes en: {src_folder}")
        return

    print(f"\n{'─'*50}")
    print(f"Clase: {class_name}")
    print(f"  Originales encontradas : {n_originals}")
    print(f"  Objetivo total         : {TARGET}")
    print(f"  A generar              : {max(0, TARGET - n_originals)}")

    # 1. Copiar originales al destino
    print("  Copiando originales...")
    for i, path in enumerate(tqdm(image_paths, desc="  Originales")):
        img = np.array(Image.open(path).convert("RGB"))
        out_name = f"orig_{i:04d}{IMG_EXT}"
        save_image(img, os.path.join(dst_folder, out_name))

    # 2. Generar augmentadas hasta llegar al objetivo
    to_generate = TARGET - n_originals
    if to_generate <= 0:
        print("  Ya se alcanzó el objetivo con los originales.")
        return

    aug_count = 0
    cycle = 0   # ciclo para iterar sobre las imágenes de forma circular

    print("  Generando augmentadas...")
    with tqdm(total=to_generate, desc="  Augmentadas") as pbar:
        while aug_count < to_generate:
            # Iterar de forma cíclica sobre las imágenes originales
            path = image_paths[cycle % n_originals]
            cycle += 1

            img = np.array(Image.open(path).convert("RGB"))
            augmented = augment(image=img)["image"]

            out_name = f"aug_{aug_count:05d}{IMG_EXT}"
            save_image(augmented, os.path.join(dst_folder, out_name))

            aug_count += 1
            pbar.update(1)

    total_saved = len(os.listdir(dst_folder))
    print(f"  Total guardadas en '{dst_folder}': {total_saved}")


# ─────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  DATA AUGMENTATION — Detección de Escoliosis")
    print("=" * 50)
    print(f"  Origen  : {INPUT_DIR}")
    print(f"  Destino : {OUTPUT_DIR}")
    print(f"  Objetivo: {TARGET} imágenes por clase")

    for cls in CLASSES:
        augment_class(cls)

    print("\n" + "=" * 50)
    print("  Proceso completado.")
    print("  Usa la carpeta 'dataset_augmented' para entrenar el modelo.")
    print("=" * 50)
