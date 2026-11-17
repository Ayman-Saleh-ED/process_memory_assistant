"""Turn the four Rieth RData files (all 20 faults) into window features.

1. Download the four files from Harvard Dataverse (doi:10.7910/DVN/6C3JR1) into  Prototype/full_data/
     TEP_FaultFree_Training.RData   TEP_Faulty_Training.RData
     TEP_FaultFree_Testing.RData    TEP_Faulty_Testing.RData
2. pip install pyreadr
3. python prepare_full_data.py

Output (in full_data/): features_train.npz and features_test.npz
Each window = 20 samples (1 hour). Features per variable: mean, standard deviation, slope.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 20   # samples per window (20 x 3 min = 1 hour)
STEP = 10     # the window moves 10 samples at a time
FOLDER = Path("full_data")

# file name, and the sample where the fault starts (training: after 1 hour, testing: after 8 hours)
FILES = {
    "train": [("TEP_FaultFree_Training.RData", 20), ("TEP_Faulty_Training.RData", 20)],
    "test": [("TEP_FaultFree_Testing.RData", 160), ("TEP_Faulty_Testing.RData", 160)],
}


def window_features(values):
    """values: (samples, variables) -> one row of features per window, and the window starts."""
    windows = sliding_window_view(values, WINDOW, axis=0)[::STEP]      # (windows, variables, WINDOW)
    t = np.arange(WINDOW) - (WINDOW - 1) / 2
    mean = windows.mean(axis=2)
    std = windows.std(axis=2)
    slope = (windows * t).sum(axis=2) / (t ** 2).sum()                 # same as a straight-line fit
    starts = np.arange(0, len(values) - WINDOW + 1, STEP)
    return np.hstack([mean, std, slope]).astype(np.float32), starts


def build(df, onset):
    sensors = [c for c in df.columns if c.startswith("x")]
    X, fault, run, start = [], [], [], []
    for (f, r), g in df.groupby(["faultNumber", "simulationRun"], sort=False):
        features, starts = window_features(g.sort_values("sample")[sensors].to_numpy(dtype=np.float64))
        # windows that cross the fault start are mixed, so we skip them
        keep = ~((f > 0) & (starts < onset) & (onset < starts + WINDOW))
        # label: 0 = normal (fault-free run, or a window before the fault starts), otherwise the fault number
        label = np.where((f == 0) | (starts + WINDOW <= onset), 0, int(f))
        X.append(features[keep])
        fault.append(label[keep])
        run.append(np.full(keep.sum(), int(r)))
        start.append(starts[keep])
    return np.vstack(X), np.concatenate(fault), np.concatenate(run), np.concatenate(start)


if __name__ == "__main__":
    import pyreadr

    for split, files in FILES.items():
        parts = []
        for name, onset in files:
            print("reading", name, "...")
            table = list(pyreadr.read_r(str(FOLDER / name)).values())[0]
            parts.append(build(table, onset))
            del table
        X = np.vstack([p[0] for p in parts])
        fault = np.concatenate([p[1] for p in parts])
        run = np.concatenate([p[2] for p in parts])
        start = np.concatenate([p[3] for p in parts])
        np.savez_compressed(FOLDER / f"features_{split}.npz", X=X, fault=fault, run=run, start=start)
        print(split, X.shape, "windows per state:", dict(zip(*np.unique(fault, return_counts=True))))


