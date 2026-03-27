"""
data_utils.py — Split sin data leakage
========================================
Garantiza que ninguna imagen de test/val es una versión aumentada
de una imagen de entrenamiento.

Convención de nombres generada por data_augmentation.py:
  orig_XXXX.jpg  → imagen original en índice XXXX
  aug_YYYYY.jpg  → derivada de orig_{YYYYY % n_orig}.jpg

Reglas del split:
  Train → orig del grupo train  +  todos sus aug_*
  Val   → SOLO orig del grupo val   (sin aumentadas)
  Test  → SOLO orig del grupo test  (sin aumentadas)
"""

import os
import re
from sklearn.model_selection import train_test_split

from config import CLASSES, TEST_SPLIT, VAL_SPLIT, SEED


def grouped_split(dataset_dir,
                  test_split=TEST_SPLIT,
                  val_split=VAL_SPLIT,
                  seed=SEED):
    """
    Construye un split train/val/test sin data leakage.

    Args:
        dataset_dir: Carpeta raíz con subcarpetas por clase
                     (scoliosis_no/, scoliosis_yes/).
        test_split:  Fracción del total original para test.
        val_split:   Fracción del total original para val.
        seed:        Semilla para reproducibilidad.

    Returns:
        dict con claves "train", "val", "test".
        Cada valor es una lista de tuplas (ruta_absoluta, label_int).
        label_int: 0 = scoliosis_no, 1 = scoliosis_yes.

    Ejemplo:
        splits = grouped_split(dataset_dir)
        train_paths, train_labels = zip(*splits["train"])
    """
    all_splits = {"train": [], "val": [], "test": []}

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No se encontró la carpeta: {class_dir}")

        all_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(".jpg")]

        orig_files = sorted([f for f in all_files if f.startswith("orig_")])
        aug_files  = sorted([f for f in all_files if f.startswith("aug_")])
        n_orig = len(orig_files)

        if n_orig == 0:
            raise ValueError(
                f"No se encontraron imágenes 'orig_*.jpg' en {class_dir}. "
                f"Asegúrate de haber ejecutado data_augmentation.py."
            )

        # ── Split estratificado sobre los originales ──────────
        indices = list(range(n_orig))
        # Separar test primero
        idx_trainval, idx_test = train_test_split(
            indices,
            test_size=test_split,
            random_state=seed,
            shuffle=True,
        )
        # Separar val del resto
        val_fraction = val_split / (1 - test_split)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )

        train_set = set(idx_train)
        val_set   = set(idx_val)
        test_set  = set(idx_test)

        # ── Asignar orig_* a su split ─────────────────────────
        for i, fname in enumerate(orig_files):
            path = os.path.join(class_dir, fname)
            if i in train_set:
                all_splits["train"].append((path, label_idx))
            elif i in val_set:
                all_splits["val"].append((path, label_idx))
            else:
                all_splits["test"].append((path, label_idx))

        # ── Asignar aug_* SOLO al train de su original ────────
        for fname in aug_files:
            m = re.search(r"aug_(\d+)", fname)
            if not m:
                continue
            aug_idx = int(m.group(1))
            source_orig_idx = aug_idx % n_orig
            if source_orig_idx in train_set:
                path = os.path.join(class_dir, fname)
                all_splits["train"].append((path, label_idx))
            # Si el original fuente está en val o test, descartamos el aug

        _print_split_summary(class_name, idx_train, idx_val, idx_test,
                              aug_files, n_orig, train_set)

    return all_splits


def _print_split_summary(class_name, idx_train, idx_val, idx_test,
                          aug_files, n_orig, train_set):
    aug_in_train = sum(
        1 for f in aug_files
        if (lambda m: m and int(m.group(1)) % n_orig in train_set)(
            re.search(r"aug_(\d+)", f)
        )
    )
    print(
        f"  {class_name:<20} "
        f"train={len(idx_train)} orig + {aug_in_train} aug  |  "
        f"val={len(idx_val)} orig  |  "
        f"test={len(idx_test)} orig"
    )
