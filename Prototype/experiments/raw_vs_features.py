"""Is the similar-episode search better on the raw sensor window than on the 156 summary features?

Run from the Prototype folder (needs the 4 RData files in full_data/):
    python experiments/raw_vs_features.py            # a few minutes

Memory = windows from training runs 1-20 (like the app). Queries = windows from testing runs 1-10 (never in the memory).
We measure: is the closest past window of the same state (top 1), and is at least one of the 5 closest of the same state (top 5)?
Saves results/raw_vs_features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import EXCLUDED, read_rdata, window_labels

WINDOW, STEP = 20, 10


def raw_windows(table, onset, max_run, mean, std):
    sensors = [c for c in table.columns if c.startswith("x")]
    table = table[(table.simulationRun <= max_run) & ~table.faultNumber.isin(EXCLUDED)]
    raw, feat, label = [], [], []
    for (f, r), g in table.groupby(["faultNumber", "simulationRun"], sort=False):
        v = g.sort_values("sample")[sensors].to_numpy(dtype=np.float64)
        z = (v - mean) / std                                          # every variable in units of normal spread
        starts = np.arange(0, len(v) - WINDOW + 1, STEP)
        lab, keep = window_labels(int(f), starts, WINDOW, onset)
        wins = np.stack([z[s:s + WINDOW] for s in starts])             # (windows, WINDOW, variables)
        t = np.arange(WINDOW) - (WINDOW - 1) / 2
        feats = np.hstack([wins.mean(1), wins.std(1), (wins * t[None, :, None]).sum(1) / (t ** 2).sum()])
        raw.append(wins.reshape(len(wins), -1)[keep]); feat.append(feats[keep]); label.append(lab[keep])
    return np.vstack(raw), np.vstack(feat), np.concatenate(label)


def score(name, memory, mem_y, query, q_y, metric):
    nn = NearestNeighbors(n_neighbors=5, metric=metric, algorithm="brute").fit(memory)
    idx = nn.kneighbors(query, return_distance=False)
    neighbour = mem_y[idx]
    return dict(method=name, top1=round(float((neighbour[:, 0] == q_y).mean()), 4),
                top5=round(float((neighbour == q_y[:, None]).any(axis=1).mean()), 4))


if __name__ == "__main__":
    normal = read_rdata("TEP_FaultFree_Training.RData")
    sensors = [c for c in normal.columns if c.startswith("x")]
    mean, std = normal[sensors].mean().to_numpy(), normal[sensors].std().to_numpy()
    faulty = read_rdata("TEP_Faulty_Training.RData")
    mem_raw, mem_feat, mem_y = raw_windows(pd.concat([normal, faulty]), 20, 20, mean, std)
    del normal, faulty
    test = pd.concat([read_rdata("TEP_FaultFree_Testing.RData"), read_rdata("TEP_Faulty_Testing.RData")])
    q_raw, q_feat, q_y = raw_windows(test, 160, 10, mean, std)
    print("memory:", mem_raw.shape, "queries:", q_raw.shape)

    scaler = StandardScaler().fit(mem_feat)
    rows = [
        score("summary features (what the app uses), cosine", scaler.transform(mem_feat), mem_y, scaler.transform(q_feat), q_y, "cosine"),
        score("raw window, cosine", mem_raw, mem_y, q_raw, q_y, "cosine"),
        score("raw window, euclidean distance", mem_raw, mem_y, q_raw, q_y, "euclidean"),
    ]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/raw_vs_features.csv", index=False)
