"""Train a gradient boosting classifier on normal operation + the 17 detectable faults.

Faults 3, 9 and 15 are left out: with these sensors they can not be told apart from normal
operation (see the first training run: the model called half of all normal windows 3, 9 or 15).
We report them separately instead.

Run from the Prototype folder after prepare_full_data.py:
    python train_models.py

Train on runs 1-400 of the training file, pick the best settings on runs 401-500,
and test once on the separate testing file.
Saves: models/gb_model.joblib, models/scaler.joblib, models/train_info.json, results/*
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("full_data")
EXCLUDED = [3, 9, 15]

# settings to compare (learning rate, size of each tree, number of boosting rounds)
CONFIGS = [
    dict(learning_rate=0.1, max_leaf_nodes=31, max_iter=200),
    dict(learning_rate=0.1, max_leaf_nodes=63, max_iter=200),
    dict(learning_rate=0.05, max_leaf_nodes=63, max_iter=400),
    dict(learning_rate=0.1, max_leaf_nodes=127, max_iter=150),
]


Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

train = np.load(FOLDER / "features_train.npz")
test = np.load(FOLDER / "features_test.npz")
X_all, y_all, run_all = train["X"], train["fault"], train["run"]
X_test_all, y_test_all = test["X"], test["fault"]

# leave out faults 3, 9, 15 (kept aside for the report at the end)
keep = ~np.isin(y_all, EXCLUDED)
keep_test = ~np.isin(y_test_all, EXCLUDED)
X_hard, y_hard = X_test_all[~keep_test], y_test_all[~keep_test]
X_all, y_all, run_all = X_all[keep], y_all[keep], run_all[keep]
X_test, y_test = X_test_all[keep_test], y_test_all[keep_test]

# split by run, so windows from the same run never end up on both sides
is_val = run_all > 0.8 * run_all.max()
X_train, y_train = X_all[~is_val], y_all[~is_val]
X_val, y_val = X_all[is_val], y_all[is_val]
print("train", X_train.shape, "| validation", X_val.shape, "| test", X_test.shape)


def scores(model, X, y):
    pred = model.predict(X)
    return pred, accuracy_score(y, pred), f1_score(y, pred, average="macro")


# ---- reference: logistic regression (the baseline from the prototype) ----
scaler = StandardScaler().fit(X_train)
start = time.time()
lr = LogisticRegression(max_iter=500).fit(scaler.transform(X_train), y_train)
print(f"logistic regression trained in {time.time() - start:.0f}s")
_, acc, f1 = scores(lr, scaler.transform(X_val), y_val)
print(f"LogReg   validation: accuracy {acc:.3f}, macro F1 {f1:.3f}")
lr_test_pred, acc, f1 = scores(lr, scaler.transform(X_test), y_test)
print(f"LogReg   test      : accuracy {acc:.3f}, macro F1 {f1:.3f}")
lr_test = (acc, f1)

# ---- gradient boosting: try the settings, keep the best on the validation runs ----
results, best_model, best_f1 = [], None, -1
for cfg in CONFIGS:
    start = time.time()
    model = HistGradientBoostingClassifier(random_state=0, **cfg).fit(X_train, y_train)
    _, acc, f1 = scores(model, X_val, y_val)
    seconds = time.time() - start
    print(f"{cfg}: validation accuracy {acc:.3f}, macro F1 {f1:.3f}  ({seconds:.0f}s)")
    results.append({**cfg, "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4), "seconds": round(seconds)})
    if f1 > best_f1:
        best_model, best_f1, best_cfg = model, f1, cfg
pd.DataFrame(results).to_csv("results/gb_settings.csv", index=False)
print("best settings:", best_cfg)

gb = best_model
gb_test_pred, gb_acc, gb_f1 = scores(gb, X_test, y_test)
print(f"GradBoost test: accuracy {gb_acc:.3f}, macro F1 {gb_f1:.3f}")

# ---- per-state results on the test file ----
classes = [int(c) for c in np.unique(y_test)]
gb_rep = classification_report(y_test, gb_test_pred, output_dict=True, zero_division=0)
lr_rep = classification_report(y_test, lr_test_pred, output_dict=True, zero_division=0)
table = pd.DataFrame({
    "precision": [gb_rep[str(c)]["precision"] for c in classes],
    "recall": [gb_rep[str(c)]["recall"] for c in classes],
    "f1": [gb_rep[str(c)]["f1-score"] for c in classes],
    "support": [gb_rep[str(c)]["support"] for c in classes],
    "recall_logreg": [lr_rep[str(c)]["recall"] for c in classes],
}, index=pd.Index(classes, name="state (0 = normal)"))
print(table.round(3).to_string())
table.round(4).to_csv("results/per_fault_test.csv")

# ---- the left-out faults: what does the model say about them? ----
if len(y_hard):
    hard_pred = gb.predict(X_hard)
    print("Left-out faults, share of windows called normal (0):")
    for f in EXCLUDED:
        m = y_hard == f
        if m.any():
            print(f"  fault {f}: {(hard_pred[m] == 0).mean():.2f}")
    # if a left-out fault is called normal, the model could not see it
    # also: how often are normal windows called a fault?
normal = y_test == 0
print(f"Normal windows called normal: {(gb_test_pred[normal] == 0).mean():.3f}")

# ---- confusion matrix (each row adds up to 1) ----
cm = confusion_matrix(y_test, gb_test_pred, labels=classes, normalize="true")
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("Real")
ax.set_title("Gradient boosting on the testing file (0 = normal)")
fig.colorbar(im)
fig.tight_layout()
fig.savefig("results/confusion_matrix_gb.png", dpi=150)

joblib.dump(gb, "models/gb_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
json.dump({"classes": classes, "excluded": EXCLUDED, "sklearn": sklearn.__version__, "best_settings": best_cfg,
           "test_accuracy": round(gb_acc, 4), "test_macro_f1": round(gb_f1, 4),
           "logreg_test_accuracy": round(lr_test[0], 4), "logreg_test_macro_f1": round(lr_test[1], 4)},
          open("models/train_info.json", "w"), indent=1)
np.savez_compressed("results/test_predictions.npz", true=y_test, gb=gb_test_pred, logreg=lr_test_pred)
print("saved models/ and results/")
print(f"If you deploy: put  scikit-learn=={sklearn.__version__}  in requirements.txt")
