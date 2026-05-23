import pickle, numpy as np
for name, path in [
    ("E1 CNN",    "outputs/robustness/robustness_cnn/robustness_results.pkl"),
    ("E2 Pose",   "outputs/robustness/robustness_pose/robustness_results.pkl"),
    ("E3 Hybrid", "outputs/robustness/robustness_hybrid/robustness_results.pkl"),
]:
    with open(path, "rb") as f: r = pickle.load(f)
    print(f"\n{name}")
    for sigma, folds in r.items():
        f1  = np.mean([d["f1"]  for d in folds])
        auc = np.mean([d["auc"] for d in folds if not np.isnan(d["auc"])])
        print(f"  σ={sigma:.2f}  F1={f1:.4f}  AUC={auc:.4f}")