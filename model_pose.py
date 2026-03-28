"""
Enfoque 2 — Pose Estimation + Features Clínicas + Clasificador ML
==================================================================
Pipeline:
  1. MediaPipe Pose extrae 33 keypoints por imagen
  2. Se calculan 12 features geométricas clínicamente relevantes
     (asimetría de hombros, caderas, inclinación del tronco, etc.)
  3. Se entrenan y comparan 3 clasificadores: RandomForest, XGBoost, SVM
  4. SHAP values para explicar qué feature impulsa cada predicción

Uso:
    from model_pose import PoseClassifier
    clf = PoseClassifier()
    clf.extract_features(dataset_dir)
    clf.train()
    clf.evaluate()
    clf.explain()
"""

import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import xgboost as xgb
import shap

from data_utils import _norm
from config import (
    CLASSES, POSE_FEATURE_NAMES, NUM_POSE_FEATURES,
    MP_LANDMARKS, OUTPUT_DIR, SEED, TEST_SPLIT, VAL_SPLIT
)


# ─────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE MEDIAPIPE (nueva API Tasks, ≥0.10.14)
# ─────────────────────────────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_MODEL_LOCAL = "pose_landmarker_full.task"


def _get_landmarker():
    """
    Descarga el modelo la primera vez y crea un PoseLandmarker.
    Retorna el landmarker listo para usar (modo IMAGE estático).
    """
    if not os.path.exists(_MODEL_LOCAL):
        print(f"Descargando modelo MediaPipe Pose (~10 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_LOCAL)
        print("  Descarga completada.")

    BaseOptions         = mp.tasks.BaseOptions
    PoseLandmarker      = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOpts  = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode         = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=_MODEL_LOCAL),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES CON MEDIAPIPE
# ─────────────────────────────────────────────────────────────

def extract_pose_features_from_image(image_path, landmarker):
    """
    Extrae features geométricas clínicas de una imagen usando MediaPipe Pose.

    Args:
        image_path: ruta a la imagen JPG.
        landmarker: instancia de mp.tasks.vision.PoseLandmarker.

    Retorna: np.array de shape (NUM_POSE_FEATURES,) o None si falla detección.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    lm  = result.pose_landmarks[0]   # landmarks de la primera persona detectada
    idx = MP_LANDMARKS

    # Extraer coordenadas normalizadas (0-1) de los 8 landmarks clave
    ls = lm[idx["left_shoulder"]]
    rs = lm[idx["right_shoulder"]]
    le = lm[idx["left_elbow"]]
    re = lm[idx["right_elbow"]]
    lw = lm[idx["left_wrist"]]
    rw = lm[idx["right_wrist"]]
    lh = lm[idx["left_hip"]]
    rh = lm[idx["right_hip"]]

    # ── Features (todas normalizadas 0-1 por MediaPipe) ───────

    # Diferencias de altura (y crece hacia abajo en imagen)
    shoulder_height_diff = abs(ls.y - rs.y)
    hip_height_diff      = abs(lh.y - rh.y)
    elbow_height_diff    = abs(le.y - re.y)
    wrist_height_diff    = abs(lw.y - rw.y)

    # Anchos
    shoulder_width = abs(ls.x - rs.x) + 1e-8   # evitar div/0
    hip_width      = abs(lh.x - rh.x) + 1e-8

    # Ratios de asimetría (independientes de la escala)
    shoulder_asym_ratio = shoulder_height_diff / shoulder_width
    hip_asym_ratio      = hip_height_diff / hip_width

    # Línea media: midpoint de hombros y caderas
    mid_shoulder_x = (ls.x + rs.x) / 2
    mid_shoulder_y = (ls.y + rs.y) / 2
    mid_hip_x      = (lh.x + rh.x) / 2
    mid_hip_y      = (lh.y + rh.y) / 2

    # Inclinación del tronco: desplazamiento horizontal midHip→midShoulder
    trunk_tilt_x  = mid_shoulder_x - mid_hip_x

    # Ángulo de inclinación del tronco (en grados)
    dy = mid_shoulder_y - mid_hip_y + 1e-8
    dx = trunk_tilt_x
    trunk_tilt_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    # Desviación de la columna (proxy): diferencia x entre línea media
    # de hombros y caderas (debería ser ~0 en columna recta)
    spine_deviation = abs(trunk_tilt_x)

    # Altura del cuerpo (proxy): distancia vertical shoulder → hip
    body_height_proxy = abs(mid_shoulder_y - mid_hip_y)

    features = np.array([
        shoulder_height_diff,
        hip_height_diff,
        elbow_height_diff,
        wrist_height_diff,
        shoulder_width,
        hip_width,
        shoulder_asym_ratio,
        hip_asym_ratio,
        trunk_tilt_x,
        trunk_tilt_angle,
        spine_deviation,
        body_height_proxy,
    ], dtype=np.float32)

    return features


def _extract_for_pairs(split_pairs, landmarker, split_name="", verbose=True):
    """
    Extrae features de pose para una lista de (path, label) pares.
    Retorna (X, y, skipped).
    """
    X, y = [], []
    skipped = 0
    if verbose:
        print(f"  Extrayendo {split_name}: {len(split_pairs)} imágenes...")

    for img_path, label in split_pairs:
        feats = extract_pose_features_from_image(str(img_path), landmarker)
        if feats is None:
            skipped += 1
            continue
        X.append(feats)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), skipped


def extract_and_split_features(dataset_dir, save_dir=None,
                                test_split=TEST_SPLIT, val_split=VAL_SPLIT,
                                seed=SEED, verbose=True):
    """
    Extrae features de pose con split sin data leakage.

    Flujo:
      1. grouped_split() determina qué imágenes van a train/val/test
      2. MediaPipe extrae features SOLO para los paths de cada split
      → No hay comparación de paths, no hay riesgo de leakage por paths

    Args:
        dataset_dir: Carpeta raíz del dataset augmentado.
        save_dir:    Si se indica, guarda los resultados en split_features.pkl.
        test_split, val_split, seed: Parámetros de split.
        verbose:     Imprimir progreso.

    Returns:
        dict con claves "train", "val", "test".
        Cada valor es {"X": np.array, "y": np.array}.
    """
    from data_utils import grouped_split

    print("Construyendo splits sin data leakage...")
    splits_paths = grouped_split(dataset_dir, test_split=test_split,
                                 val_split=val_split, seed=seed)

    landmarker = _get_landmarker()
    result = {}
    total_skipped = 0

    for split_name in ["train", "val", "test"]:
        X, y, skipped = _extract_for_pairs(
            splits_paths[split_name], landmarker, split_name, verbose
        )
        result[split_name] = {"X": X, "y": y}
        total_skipped += skipped

        labels, counts = np.unique(y, return_counts=True) if len(y) > 0 else ([], [])
        dist = dict(zip(labels.tolist(), counts.tolist())) if len(labels) > 0 else {}
        print(f"    → {len(X)} features | {skipped} sin pose | dist={dist}")

    landmarker.close()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "split_features.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Features guardadas en: {save_path}")

    return result


# ─────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS (legacy — mantenido para compatibilidad)
# ─────────────────────────────────────────────────────────────

def extract_features_from_dataset(dataset_dir, save_path=None, verbose=True):
    """
    [LEGACY] Extrae features para TODAS las imágenes.
    Usar extract_and_split_features() para el pipeline sin leakage.
    """
    landmarker = _get_landmarker()

    X, y, paths = [], [], []
    skipped = 0

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[ADVERTENCIA] No existe: {class_dir}")
            continue

        img_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(".jpg")]

        if verbose:
            print(f"\nExtrayendo features: {class_name} ({len(img_files)} imgs)")

        for fname in img_files:
            img_path = os.path.join(class_dir, fname)
            feats = extract_pose_features_from_image(img_path, landmarker)

            if feats is None:
                skipped += 1
                continue

            X.append(feats)
            y.append(class_idx)
            paths.append(_norm(img_path))

    landmarker.close()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"X": X, "y": y, "paths": paths}, f)

    return X, y, paths, skipped


def split_data(X, y, paths, dataset_dir,
               test_split=TEST_SPLIT, val_split=VAL_SPLIT, seed=SEED):
    """[LEGACY] Usa extract_and_split_features() en su lugar."""
    from data_utils import grouped_split

    splits = grouped_split(dataset_dir, test_split=test_split,
                           val_split=val_split, seed=seed)

    train_paths = {p for p, _ in splits["train"]}
    val_paths   = {p for p, _ in splits["val"]}
    test_paths  = {p for p, _ in splits["test"]}

    def _filter(path_set):
        idx = [i for i, p in enumerate(paths) if _norm(p) in path_set]
        return X[idx], y[idx]

    X_train, y_train = _filter(train_paths)
    X_val,   y_val   = _filter(val_paths)
    X_test,  y_test  = _filter(test_paths)

    print(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────
# DEFINICIÓN Y ENTRENAMIENTO DE CLASIFICADORES
# ─────────────────────────────────────────────────────────────

def build_classifiers(seed=SEED):
    """
    Retorna un dict con los 3 clasificadores a comparar,
    cada uno envuelto en un Pipeline con StandardScaler.
    """
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
    }


def train_pose_classifiers(X_train, y_train, X_val, y_val,
                            output_dir=None, seed=SEED):
    """
    Entrena los 3 clasificadores, evalúa en validación con CV,
    y retorna el mejor modelo junto con las métricas.

    Returns:
        best_name: nombre del mejor clasificador
        best_model: pipeline entrenado
        all_models: dict con todos los pipelines entrenados
        cv_results: dict con scores de CV
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "pose")
    os.makedirs(output_dir, exist_ok=True)

    classifiers = build_classifiers(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    cv_results = {}
    for name, pipe in classifiers.items():
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:15s} CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
        pipe.fit(X_train, y_train)

    # Elegir el mejor según AUC en validación
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_model = classifiers[best_name]

    print(f"\nMejor clasificador: {best_name}")

    # Guardar todos los modelos
    for name, pipe in classifiers.items():
        model_path = os.path.join(output_dir, f"pose_{name.lower()}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
    print(f"Modelos guardados en: {output_dir}")

    # Graficar comparativa CV
    _plot_cv_comparison(cv_results, output_dir)

    return best_name, best_model, classifiers, cv_results


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────

def evaluate_pose_classifiers(classifiers, X_test, y_test, output_dir=None):
    """
    Evalúa todos los clasificadores en test y genera gráficas comparativas.

    Returns:
        results: dict {model_name: {accuracy, auc, report, ...}}
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "pose")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    plt.figure(figsize=(8, 6))

    for name, pipe in classifiers.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        acc = np.mean(y_test == y_pred)

        results[name] = {"accuracy": acc, "auc": auc,
                         "y_prob": y_prob, "y_pred": y_pred}

        print(f"\n{'─'*45}")
        print(f"  {name}")
        print(f"{'─'*45}")
        print(classification_report(y_test, y_pred,
                                    target_names=["scoliosis_no", "scoliosis_yes"]))
        print(f"AUC-ROC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Curvas ROC — Clasificadores de Pose")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pose_roc_comparison.png"), dpi=150)
    plt.show()

    # Confusion matrix del mejor (mayor AUC)
    best = max(results, key=lambda k: results[k]["auc"])
    _plot_confusion_matrix_pose(y_test, results[best]["y_pred"],
                                output_dir, model_name=best)

    return results


# ─────────────────────────────────────────────────────────────
# EXPLICABILIDAD — SHAP VALUES
# ─────────────────────────────────────────────────────────────

def explain_with_shap(pipeline, X_test, feature_names=None, output_dir=None):
    """
    Calcula e imprime SHAP values para el clasificador ML.
    Genera: beeswarm plot y bar plot de importancia media.

    Nota: Para RandomForest y XGBoost usa TreeExplainer (rápido).
          Para SVM usa KernelExplainer (lento, usa muestra pequeña).
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "pose")
    os.makedirs(output_dir, exist_ok=True)

    if feature_names is None:
        feature_names = POSE_FEATURE_NAMES

    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X_test)

    clf_type = type(clf).__name__

    if clf_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_scaled)
        # RandomForest devuelve lista [clase0, clase1]; XGBoost devuelve array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # clase scoliosis_yes
    else:
        # SVM: usar un subconjunto pequeño para KernelExplainer
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer = shap.KernelExplainer(
            lambda x: pipeline.predict_proba(
                scaler.inverse_transform(x)
            )[:, 1],
            background
        )
        shap_values = explainer.shap_values(X_scaled[:50])
        X_scaled = X_scaled[:50]

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=feature_names, show=False)
    plt.title(f"SHAP Beeswarm — {clf_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{clf_type}.png"), dpi=150)
    plt.show()

    # Bar plot (media absoluta)
    plt.figure()
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Importancia media — {clf_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_bar_{clf_type}.png"), dpi=150)
    plt.show()

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nImportancia de features (SHAP):")
    print(importance_df.to_string(index=False))
    importance_df.to_csv(
        os.path.join(output_dir, f"shap_importance_{clf_type}.csv"), index=False
    )
    return importance_df


# ─────────────────────────────────────────────────────────────
# GRÁFICAS AUXILIARES
# ─────────────────────────────────────────────────────────────

def _plot_cv_comparison(cv_results, output_dir):
    names  = list(cv_results.keys())
    means  = [cv_results[n].mean() for n in names]
    stds   = [cv_results[n].std()  for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=6,
                   color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.ylabel("AUC-ROC (CV 5-fold)"); plt.title("Comparativa de clasificadores — Pose")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pose_cv_comparison.png"), dpi=150)
    plt.show()


def _plot_confusion_matrix_pose(y_true, y_pred, output_dir, model_name=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["no", "yes"],
                yticklabels=["no", "yes"], ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {model_name} (Pose)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                f"pose_{model_name.lower()}_confusion.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA DIRECTO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DATASET_AUG_DIR

    out = os.path.join(OUTPUT_DIR, "pose")
    features_path = os.path.join(out, "pose_features.pkl")

    # 1. Extraer o cargar features
    if os.path.exists(features_path):
        print(f"Cargando features desde: {features_path}")
        with open(features_path, "rb") as f:
            data = pickle.load(f)
        X, y, paths = data["X"], data["y"], data["paths"]
    else:
        X, y, paths, _ = extract_features_from_dataset(
            DATASET_AUG_DIR, save_path=features_path
        )

    # 2. Split sin data leakage
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, paths, DATASET_AUG_DIR
    )

    # 3. Entrenar
    best_name, best_model, all_models, cv_res = train_pose_classifiers(
        X_train, y_train, X_val, y_val, output_dir=out
    )

    # 4. Evaluar
    results = evaluate_pose_classifiers(all_models, X_test, y_test, output_dir=out)

    # 5. SHAP
    explain_with_shap(best_model, X_test, output_dir=out)
