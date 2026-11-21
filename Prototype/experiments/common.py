"""Small helpers shared by the experiment scripts (window features, reading the data)."""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EXCLUDED = [3, 9, 15]
FOLDER = Path("full_data")


def window_features(values, window, step, extended=False):
    """values: (samples, variables). Returns one row of features per window, and the window starts.
    basic    = mean, standard deviation, slope
    extended = basic + minimum, maximum, last value
    """
    wins = sliding_window_view(values, window, axis=0)[::step]          # (windows, variables, window)
    t = np.arange(window) - (window - 1) / 2
    parts = [wins.mean(axis=2), wins.std(axis=2), (wins * t).sum(axis=2) / (t ** 2).sum()]
    if extended:
        parts += [wins.min(axis=2), wins.max(axis=2), wins[:, :, -1]]
    starts = np.arange(0, len(values) - window + 1, step)
    return np.hstack(parts).astype(np.float32), starts


def window_labels(fault, starts, window, onset):
    """label 0 = normal (fault-free run, or window ends before the fault starts); keep = window does not cross the onset."""
    keep = ~((fault > 0) & (starts < onset) & (onset < starts + window))
    label = np.where((fault == 0) | (starts + window <= onset), 0, int(fault))
    return label, keep


def table_to_runs(table, drop=EXCLUDED):
    """DataFrame -> list of (fault, run, values float32). Faults in `drop` are left out."""
    sensors = [c for c in table.columns if c.startswith("x")]
    runs = []
    for (f, r), g in table.groupby(["faultNumber", "simulationRun"], sort=False):
        if int(f) in drop:
            continue
        runs.append((int(f), int(r), g.sort_values("sample")[sensors].to_numpy(dtype=np.float32)))
    return runs


def read_rdata(name):
    import pyreadr
    return list(pyreadr.read_r(str(FOLDER / name)).values())[0]


def build(runs, onset, window, step, extended=False):
    """All windows of all runs -> X, label, run, start (windows that cross the fault start are dropped)."""
    X, y, run, start = [], [], [], []
    for f, r, values in runs:
        feats, starts = window_features(values.astype(np.float64), window, step, extended)
        label, keep = window_labels(f, starts, window, onset)
        X.append(feats[keep]); y.append(label[keep])
        run.append(np.full(keep.sum(), r)); start.append(starts[keep])
    return np.vstack(X), np.concatenate(y), np.concatenate(run), np.concatenate(start)
