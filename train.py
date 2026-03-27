"""
Script de entrenamiento unificado
===================================
Permite entrenar cualquiera de los 3 enfoques desde la terminal
y, opcionalmente, correr el ablation study completo.

Uso:
    # Enfoque 1 — CNN
    python train.py --model cnn

    # Enfoque 2 — Pose + ML
    python train.py --model pose

    # Enfoque 3 — Híbrido
    python train.py --model hybrid

    # Ablation study completo (entrena los 3 y los compara)
    python train.py --model all

Opciones adicionales:
    --dataset   Ruta al dataset augmentado (default: config.DATASET_AUG_DIR)
    --output    Ruta base de salida          (default: config.OUTPUT_DIR)
"""

import argparse
import os
import sys

from config import DATASET_AUG_DIR, OUTPUT_DIR


# ─────────────────────────────────────────────────────────────

def train_cnn_model(dataset_dir, output_dir):
    from model_cnn import train_cnn, evaluate_cnn

    print("\n" + "█"*55)
    print("  ENFOQUE 1 — CNN Transfer Learning")
    print("█"*55)

    out = os.path.join(output_dir, "cnn")
    model, h1, h2, test_ds = train_cnn(dataset_dir, out)
    results = evaluate_cnn(model, test_ds, out)

    print(f"\n[E1 CNN] Accuracy: {results['accuracy']:.4f} | AUC: {results['auc']:.4f}")
    return model, results


def train_pose_model(dataset_dir, output_dir):
    import pickle
    from model_pose import (
        extract_features_from_dataset, split_data,
        train_pose_classifiers, evaluate_pose_classifiers,
        explain_with_shap
    )

    print("\n" + "█"*55)
    print("  ENFOQUE 2 — Pose Estimation + Clasificador ML")
    print("█"*55)

    out = os.path.join(output_dir, "pose")
    os.makedirs(out, exist_ok=True)
    features_path = os.path.join(out, "pose_features.pkl")

    # Extraer o reutilizar features
    if os.path.exists(features_path):
        print(f"  Cargando features de pose existentes: {features_path}")
        with open(features_path, "rb") as f:
            data = pickle.load(f)
        X, y, paths = data["X"], data["y"], data["paths"]
    else:
        print("  Extrayendo features de pose (puede tardar varios minutos)...")
        X, y, paths, skipped = extract_features_from_dataset(
            dataset_dir, save_path=features_path
        )
        print(f"  Imágenes saltadas (sin pose detectada): {skipped}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, paths, dataset_dir
    )

    best_name, best_model, all_models, cv_res = train_pose_classifiers(
        X_train, y_train, X_val, y_val, output_dir=out
    )

    results_all = evaluate_pose_classifiers(all_models, X_test, y_test, output_dir=out)
    explain_with_shap(best_model, X_test, output_dir=out)

    # Retornar resultados del mejor modelo en formato compatible
    best_res = results_all[best_name]
    print(f"\n[E2 Pose/{best_name}] Accuracy: {best_res['accuracy']:.4f} | "
          f"AUC: {best_res['auc']:.4f}")

    return best_model, best_res, results_all


def train_hybrid_model(dataset_dir, output_dir):
    from model_hybrid import train_hybrid, evaluate_hybrid

    print("\n" + "█"*55)
    print("  ENFOQUE 3 — Modelo Híbrido (CNN + Pose)")
    print("█"*55)

    out = os.path.join(output_dir, "hybrid")
    features_path = os.path.join(output_dir, "pose", "pose_features.pkl")

    if not os.path.exists(features_path):
        print("\n  ERROR: Las features de pose no existen.")
        print("  Ejecuta primero: python train.py --model pose")
        sys.exit(1)

    model, h1, h2, test_ds = train_hybrid(dataset_dir, features_path, out)
    results = evaluate_hybrid(model, test_ds, out)

    print(f"\n[E3 Híbrido] Accuracy: {results['accuracy']:.4f} | "
          f"AUC: {results['auc']:.4f}")
    return model, results


def run_ablation(dataset_dir, output_dir):
    """Entrena los 3 modelos y genera el ablation study comparativo."""
    from model_hybrid import ablation_study

    _, results_cnn                         = train_cnn_model(dataset_dir, output_dir)
    _, results_pose_best, _                = train_pose_model(dataset_dir, output_dir)
    _, results_hybrid                      = train_hybrid_model(dataset_dir, output_dir)

    ablation_study(
        results_cnn,
        results_pose_best,
        results_hybrid,
        output_dir=os.path.join(output_dir, "ablation")
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos de detección de escoliosis"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "pose", "hybrid", "all"],
        required=True,
        help="Modelo a entrenar: cnn | pose | hybrid | all (ablation)"
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_AUG_DIR,
        help=f"Ruta al dataset augmentado (default: {DATASET_AUG_DIR})"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"Directorio de salida (default: {OUTPUT_DIR})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*55)
    print("  Detección de Escoliosis — Sistema de IA")
    print("="*55)
    print(f"  Modelo    : {args.model.upper()}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Salida    : {args.output}")

    if args.model == "cnn":
        train_cnn_model(args.dataset, args.output)

    elif args.model == "pose":
        train_pose_model(args.dataset, args.output)

    elif args.model == "hybrid":
        train_hybrid_model(args.dataset, args.output)

    elif args.model == "all":
        run_ablation(args.dataset, args.output)

    print("\n  Entrenamiento completado.")
    print(f"  Resultados guardados en: {args.output}")
