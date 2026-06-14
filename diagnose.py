"""
diagnose.py — Herramienta de diagnóstico del pipeline
========================================================
Ejecuta esto ANTES de train.py para verificar que el split
sin data leakage funciona correctamente.

Uso:
    python diagnose.py
"""

import os
import sys
from config import DATASET_AUG_DIR, CLASSES
from data_utils import grouped_split, _norm


def diagnose_split(dataset_dir):
    print("=" * 60)
    print("  DIAGNÓSTICO DE SPLIT — Detección de Data Leakage")
    print("=" * 60)
    print(f"\nDataset: {dataset_dir}\n")

    # 1. Verificar que existe la carpeta
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] La carpeta no existe: {dataset_dir}")
        print("  Asegúrate de haber ejecutado data_augmentation.py primero.")
        sys.exit(1)

    # 2. Verificar estructura de nombres
    for class_name in CLASSES:
        class_dir = os.path.join(dataset_dir, class_name)
        all_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(".jpg")]
        orig = [f for f in all_files if f.startswith("orig_")]
        aug  = [f for f in all_files if f.startswith("aug_")]
        other = [f for f in all_files if not f.startswith("orig_")
                 and not f.startswith("aug_")]

        print(f"Clase '{class_name}':")
        print(f"  Total JPG: {len(all_files)}")
        print(f"  orig_*.jpg: {len(orig)}")
        print(f"  aug_*.jpg:  {len(aug)}")
        if other:
            print(f"  [PROBLEMA] Archivos con nombre INESPERADO ({len(other)}): "
                  f"{other[:5]}{'...' if len(other) > 5 else ''}")
            print("  → grouped_split no funcionará hasta que estos archivos")
            print("    sean renombrados o el dataset sea regenerado.")
        print()

    # 3. Ejecutar grouped_split y verificar
    print("Ejecutando grouped_split()...")
    try:
        splits = grouped_split(dataset_dir)
    except Exception as e:
        print(f"[ERROR] grouped_split falló: {e}")
        sys.exit(1)

    print()
    for split_name in ["train", "val", "test"]:
        pairs = splits[split_name]
        aug_in_split  = [p for p, _ in pairs if "aug_"  in os.path.basename(p)]
        orig_in_split = [p for p, _ in pairs if "orig_" in os.path.basename(p)]
        other_in_split= [p for p, _ in pairs
                         if "aug_" not in os.path.basename(p)
                         and "orig_" not in os.path.basename(p)]

        ok = (split_name == "train") or (len(aug_in_split) == 0)
        status = "OK" if ok else "LEAKAGE DETECTADO"

        print(f"  {split_name.upper()} [{status}]:")
        print(f"    Total: {len(pairs)} | orig: {len(orig_in_split)} | aug: {len(aug_in_split)}")
        if other_in_split:
            print(f"    [PROBLEMA] Archivos sin prefijo: {len(other_in_split)}")
        if split_name in ("val", "test") and aug_in_split:
            print(f"    [LEAKAGE] Archivos aug_* en {split_name} (primeros 3):")
            for p in aug_in_split[:3]:
                print(f"      {p}")

    print()
    print("Primeras 3 rutas del TEST set:")
    for p, lbl in splits["test"][:3]:
        print(f"  [{CLASSES[lbl]}] {p}")

    print()
    print("Verificación de normalización de paths:")
    sample_path = splits["test"][0][0] if splits["test"] else None
    if sample_path:
        print(f"  Path raw:  {repr(sample_path)}")
        print(f"  _norm():   {repr(_norm(sample_path))}")

    print()
    # 4. Verificar si hay pkl cacheado desactualizado
    old_pkl = os.path.join(
        os.path.dirname(DATASET_AUG_DIR), "outputs", "pose", "pose_features.pkl"
    )
    if os.path.exists(old_pkl):
        import pickle
        with open(old_pkl, "rb") as f:
            data = pickle.load(f)
        cached_paths = data.get("paths", [])
        print(f"PKL cacheado encontrado: {old_pkl}")
        print(f"  Imágenes en pkl: {len(cached_paths)}")
        if cached_paths:
            print(f"  Primer path en pkl: {repr(cached_paths[0])}")
            print(f"  _norm(pkl path):    {repr(_norm(cached_paths[0]))}")
            # Comparar con grouped_split
            split_set = {p for p, _ in splits["train"]}
            matches = sum(1 for p in cached_paths if _norm(p) in split_set)
            print(f"  Matches pkl→split_train: {matches}/{len(cached_paths)}")
            if matches == 0:
                print("  [PROBLEMA] NINGÚN path del pkl coincide con el split.")
                print("  → Borra el pkl y vuelve a ejecutar train.py")
    else:
        print(f"No hay pkl cacheado (correcto para primera ejecución).")

    print()
    print("=" * 60)
    print("  Si todo aparece OK arriba, el split es correcto.")
    print("  Si no, sigue las instrucciones indicadas.")
    print("=" * 60)


if __name__ == "__main__":
    diagnose_split(DATASET_AUG_DIR)
