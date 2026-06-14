"""
diagnose_features.py — Diagnóstico de separabilidad del dataset
================================================================
Extrae las features de pose de todas las imágenes originales y analiza
por qué los modelos obtienen accuracy=1.0 en este dataset.

Genera:
  - Box plots de cada feature por clase
  - Reporte de separabilidad (qué features tienen solapamiento)
  - Decision Stump para encontrar el mejor feature individual
  - Scatter plot de los 2 features más discriminativos

Uso:
    python diagnose_features.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from config import DATASET_AUG_DIR, OUTPUT_DIR, CLASSES, POSE_FEATURE_NAMES, SEED


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN
# ─────────────────────────────────────────────────────────────

def load_or_extract(dataset_dir, cache_path):
    """Extrae features de todos los orig_*.jpg o carga el cache."""
    if os.path.exists(cache_path):
        print(f"  Cargando cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from model_pose import extract_pose_features_from_image, _get_landmarker

    print("  Extrayendo features (primera vez, ~2-3 min)...")
    landmarker = _get_landmarker()

    all_features, all_labels, all_paths = [], [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        fnames = sorted([
            f for f in os.listdir(class_dir)
            if f.startswith("orig_") and f.lower().endswith(".jpg")
        ])
        print(f"  {class_name}: {len(fnames)} originales")
        for fname in fnames:
            path = os.path.join(class_dir, fname)
            feats = extract_pose_features_from_image(path, landmarker)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(label_idx)
                all_paths.append(path)
            else:
                print(f"    [WARN] Sin pose detectada: {fname}")

    landmarker.close()

    data = {
        "X": np.array(all_features, dtype=np.float32),
        "y": np.array(all_labels),
        "paths": all_paths,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Cache guardado: {cache_path}")
    return data


# ─────────────────────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────────────────────

def separability_report(X, y, feature_names):
    """
    Para cada feature, calcula:
    - Media y std por clase
    - Cohen's d (tamaño de efecto)
    - Si hay solapamiento entre clases
    """
    print("\n" + "═"*70)
    print("  REPORTE DE SEPARABILIDAD POR FEATURE")
    print("═"*70)
    print(f"  {'Feature':<28} {'no_mean':>8} {'yes_mean':>8} {'Cohen_d':>8}  {'Overlap?':>8}")
    print("  " + "─"*66)

    separabilities = []
    for i, name in enumerate(feature_names):
        vals_no  = X[y == 0, i]
        vals_yes = X[y == 1, i]

        mean_no,  std_no  = vals_no.mean(),  vals_no.std()
        mean_yes, std_yes = vals_yes.mean(), vals_yes.std()

        pooled_std = np.sqrt((std_no**2 + std_yes**2) / 2) + 1e-10
        cohens_d = abs(mean_yes - mean_no) / pooled_std

        # Solapamiento: ¿hay algún valor de yes dentro del rango de no (y viceversa)?
        min_yes, max_yes = vals_yes.min(), vals_yes.max()
        min_no,  max_no  = vals_no.min(),  vals_no.max()
        overlap = not (max_yes < min_no or max_no < min_yes)

        separabilities.append(cohens_d)
        print(f"  {name:<28} {mean_no:>8.4f} {mean_yes:>8.4f} {cohens_d:>8.2f}  {'SÍ' if overlap else 'NO':>8}")

    print("═"*70)
    best_idx = int(np.argmax(separabilities))
    print(f"\n  Feature más discriminativo: {feature_names[best_idx]}")
    print(f"  Cohen's d = {separabilities[best_idx]:.2f} (>0.8 = efecto grande)")
    return separabilities


def test_simple_classifiers(X, y, feature_names, seed=SEED):
    """Prueba clasificadores simples para confirmar separabilidad."""
    print("\n" + "═"*70)
    print("  CLASIFICADORES SIMPLES (control de separabilidad)")
    print("═"*70)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Decision Stump: un solo corte en un solo feature
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    stump_scores = cross_val_score(stump, Xs, y, cv=5, scoring="accuracy")
    print(f"  Decision Stump  (max_depth=1):  {stump_scores.mean():.4f} ± {stump_scores.std():.4f}")

    # Árbol de profundidad 3
    tree3 = DecisionTreeClassifier(max_depth=3, random_state=seed)
    tree3_scores = cross_val_score(tree3, Xs, y, cv=5, scoring="accuracy")
    print(f"  Decision Tree   (max_depth=3):  {tree3_scores.mean():.4f} ± {tree3_scores.std():.4f}")

    # Regresión logística muy regularizada (difícil de overfit)
    lr_strong = LogisticRegression(C=0.001, max_iter=1000, random_state=seed)
    lr_scores  = cross_val_score(lr_strong, Xs, y, cv=5, scoring="accuracy")
    print(f"  LogReg (C=0.001, muy regularizado): {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

    # Regresión logística normal
    lr_normal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    lr_n_scores = cross_val_score(lr_normal, Xs, y, cv=5, scoring="accuracy")
    print(f"  LogReg (C=1.0, normal):             {lr_n_scores.mean():.4f} ± {lr_n_scores.std():.4f}")

    # Mejor feature individual por Decision Stump
    best_acc, best_feat, best_thresh = 0, "", 0
    for i, name in enumerate(feature_names):
        Xi = X[:, i:i+1]
        s = DecisionTreeClassifier(max_depth=1, random_state=seed)
        sc = cross_val_score(s, Xi, y, cv=5, scoring="accuracy").mean()
        if sc > best_acc:
            best_acc = sc
            best_feat = name
    print(f"\n  Mejor feature individual: '{best_feat}' → accuracy {best_acc:.4f}")
    print("═"*70)

    return stump_scores.mean(), lr_scores.mean()


# ─────────────────────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────────────────────

def plot_feature_boxplots(X, y, feature_names, output_dir):
    """Box plots de cada feature separados por clase."""
    n_feat = len(feature_names)
    cols = 4
    rows = (n_feat + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.patch.set_facecolor("#f0f4f8")
    axes = axes.flatten()

    colors = {"scoliosis_no": "#4e9af1", "scoliosis_yes": "#e05c5c"}
    class_labels = ["scoliosis_no", "scoliosis_yes"]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        data_groups = [X[y == 0, i], X[y == 1, i]]

        bp = ax.boxplot(
            data_groups,
            labels=["No", "Sí"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], [colors["scoliosis_no"], colors["scoliosis_yes"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(name.replace("_", "\n"), fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlabel("Escoliosis", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_facecolor("#fafbfc")

    # Ocultar ejes vacíos
    for ax in axes[n_feat:]:
        ax.set_visible(False)

    patch_no  = mpatches.Patch(color=colors["scoliosis_no"],  label="scoliosis_no",  alpha=0.75)
    patch_yes = mpatches.Patch(color=colors["scoliosis_yes"], label="scoliosis_yes", alpha=0.75)
    fig.legend(handles=[patch_no, patch_yes], loc="lower right",
               fontsize=10, framealpha=0.9)

    fig.suptitle("Distribución de Features Geométricas por Clase\n(Test de Adams — Imágenes Originales)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "feature_boxplots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfica guardada: {out_path}")
    plt.show()


def plot_scatter_top2(X, y, feature_names, separabilities, output_dir):
    """Scatter plot de los 2 features más discriminativos."""
    top2 = np.argsort(separabilities)[-2:][::-1]
    i1, i2 = top2[0], top2[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_facecolor("#fafbfc")

    colors_scatter = {0: "#4e9af1", 1: "#e05c5c"}
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X[mask, i1], X[mask, i2],
            c=colors_scatter[cls],
            label=["scoliosis_no", "scoliosis_yes"][cls],
            alpha=0.6, s=40, edgecolors="white", linewidths=0.5
        )

    ax.set_xlabel(feature_names[i1].replace("_", " "), fontsize=11)
    ax.set_ylabel(feature_names[i2].replace("_", " "), fontsize=11)
    ax.set_title(f"Top-2 Features Discriminativos\n"
                 f"({feature_names[i1]} vs {feature_names[i2]})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "scatter_top2_features.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Gráfica guardada: {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(OUTPUT_DIR, "feature_diagnosis")
    os.makedirs(out_dir, exist_ok=True)

    cache = os.path.join(out_dir, "orig_features_cache.pkl")

    print("\n" + "█"*55)
    print("  DIAGNÓSTICO DE FEATURES — Dataset de Escoliosis")
    print("█"*55)

    data = load_or_extract(DATASET_AUG_DIR, cache)
    X, y = data["X"], data["y"]

    print(f"\n  Total imágenes con pose detectada: {len(y)}")
    print(f"  scoliosis_no:  {(y==0).sum()}")
    print(f"  scoliosis_yes: {(y==1).sum()}")

    separabilities = separability_report(X, y, POSE_FEATURE_NAMES)
    stump_acc, lr_acc = test_simple_classifiers(X, y, POSE_FEATURE_NAMES)

    print(f"\n  Conclusión:")
    if stump_acc >= 1.0:
        print("  → Un SOLO corte en UN SOLO feature separa perfectamente las clases.")
        print("  → El dataset contiene casos clínicamente obvios sin solapamiento.")
        print("  → Esto justifica reportar 1.0 como resultado real del dataset.")
    elif stump_acc >= 0.95:
        print(f"  → Decision Stump = {stump_acc:.3f}: el dataset es casi perfectamente separable.")
    else:
        print(f"  → Decision Stump = {stump_acc:.3f}: hay algo de dificultad real.")

    plot_feature_boxplots(X, y, POSE_FEATURE_NAMES, out_dir)
    plot_scatter_top2(X, y, POSE_FEATURE_NAMES, separabilities, out_dir)

    # Guardar reporte de texto
    report_path = os.path.join(out_dir, "separability_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DE SEPARABILIDAD DEL DATASET\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total muestras: {len(y)}\n")
        f.write(f"scoliosis_no:  {(y==0).sum()}\n")
        f.write(f"scoliosis_yes: {(y==1).sum()}\n\n")
        f.write("Cohen's d por feature:\n")
        for name, d in zip(POSE_FEATURE_NAMES, separabilities):
            f.write(f"  {name:<30} d={d:.3f}\n")
        f.write(f"\nDecision Stump (1 feature, 1 corte): {stump_acc:.4f}\n")
        f.write(f"Logistic Regression (C=0.001):        {lr_acc:.4f}\n")
    print(f"  Reporte guardado: {report_path}")
