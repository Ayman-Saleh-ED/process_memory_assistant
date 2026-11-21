"""Does a different window length (or richer features) work better than 1 hour + mean/std/slope?

Run from the Prototype folder (needs the 4 RData files in full_data/):
    python experiments/sweep_windows.py            # about 30-40 minutes, saves after every setting
    python experiments/sweep_windows.py mlp        # also trains one neural network (MLP) for comparison

Every setting uses the same split as train_models.py: runs 1-400 train, 401-500 choose, the testing file is
used once. Same gradient boosting settings for all, so only the window / features change.
Faults 3, 9 and 15 are left out. Overlap is always 50% (step = window / 2), except the last row (step 5).
Saves results/window_sweep.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from common import build, read_rdata, table_to_runs

TRAIN_ONSET, TEST_ONSET, MINUTES = 20, 160, 3
# (window, step, extended features)
SETTINGS = [(20, 10, False), (20, 10, True), (10, 5, False), (10, 5, True), (40, 20, False), (40, 20, True),
            (60, 30, False), (60, 30, True), (20, 5, False)]
GB = dict(learning_rate=0.1, max_leaf_nodes=63, max_iter=200, random_state=0)


def detection(label, run, start, pred, window):
    """Detection delay on the test runs: minutes from the fault start until the end of the first window called a fault."""
    df = pd.DataFrame({"label": label, "run": run, "start": start, "pred": pred})
    faulty = df[df.label > 0]
    n_runs = faulty.groupby(["label", "run"]).ngroups
    flagged = faulty[faulty.pred != 0].groupby(["label", "run"]).start.min()
    delay = (flagged + window - TEST_ONSET) * MINUTES
    return float(delay.median()) if len(delay) else float("nan"), len(flagged) / max(n_runs, 1)


def run_sweep(train_runs, test_runs, settings=SETTINGS, mlp=False, out="results/window_sweep.csv", gb=GB):
    rows = []
    for window, step, ext in settings:
        t0 = time.time()
        X, y, run, _ = build(train_runs, TRAIN_ONSET, window, step, ext)
        Xt, yt, rt, st = build(test_runs, TEST_ONSET, window, step, ext)
        val = run > 0.8 * run.max()
        model = HistGradientBoostingClassifier(**gb).fit(X[~val], y[~val])
        val_f1 = f1_score(y[val], model.predict(X[val]), average="macro")
        pred = model.predict(Xt)
        med, share = detection(yt, rt, st, pred, window)
        rows.append(dict(model="gradient boosting", window=window, step=step, features="extended" if ext else "basic",
                         n_features=X.shape[1], val_macro_f1=round(val_f1, 4),
                         test_accuracy=round(accuracy_score(yt, pred), 4),
                         test_macro_f1=round(f1_score(yt, pred, average="macro"), 4),
                         normal_recall=round(float((pred[yt == 0] == 0).mean()), 4),
                         median_delay_min=med, runs_flagged=round(share, 4), seconds=round(time.time() - t0)))
        print(pd.DataFrame(rows).tail(1).to_string(header=len(rows) == 1, index=False), flush=True)
        if mlp and (window, step, ext) == settings[0]:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            t0 = time.time()
            sc = StandardScaler().fit(X[~val])
            net = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=60, random_state=0)
            net.fit(sc.transform(X[~val]), y[~val])
            pred = net.predict(sc.transform(Xt))
            med, share = detection(yt, rt, st, pred, window)
            rows.append(dict(model="neural network (MLP)", window=window, step=step, features="basic",
                             n_features=X.shape[1],
                             val_macro_f1=round(f1_score(y[val], net.predict(sc.transform(X[val])), average="macro"), 4),
                             test_accuracy=round(accuracy_score(yt, pred), 4),
                             test_macro_f1=round(f1_score(yt, pred, average="macro"), 4),
                             normal_recall=round(float((pred[yt == 0] == 0).mean()), 4),
                             median_delay_min=med, runs_flagged=round(share, 4), seconds=round(time.time() - t0)))
            print(pd.DataFrame(rows).tail(1).to_string(header=False, index=False), flush=True)
        Path(out).parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("reading the data (a few minutes)...")
    train_runs = table_to_runs(read_rdata("TEP_FaultFree_Training.RData"))
    train_runs += table_to_runs(read_rdata("TEP_Faulty_Training.RData"))
    test_runs = table_to_runs(read_rdata("TEP_FaultFree_Testing.RData"))
    test_runs += table_to_runs(read_rdata("TEP_Faulty_Testing.RData"))
    print(len(train_runs), "training runs,", len(test_runs), "testing runs")
    result = run_sweep(train_runs, test_runs, mlp="mlp" in sys.argv)
    print("\n", result.to_string(index=False))
    print("saved results/window_sweep.csv")
