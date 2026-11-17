"""Make the small files the deployed app needs (the full data is too big for GitHub).

Run from the Prototype folder after prepare_full_data.py and train_models.py:
    python make_demo_files.py

Writes:
    data/demo_runs.csv           2 runs of every state from the testing file (the app's test cases)
    models/memory_index.npz      window features of 20 training runs per state (the app's "past episodes")
    models/normal_stats.npz      average and spread of every variable in normal operation
    models/app_config.json       window size, fault start in the testing file, left-out faults
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_full_data import WINDOW, STEP

FOLDER = Path("full_data")
EXCLUDED = [3, 9, 15]
DEMO_RUNS = [1, 2]    # runs taken from the testing file
INDEX_RUNS = 20       # training runs per state kept in the memory index
TEST_ONSET = 160      # the fault starts after sample 160 in the testing file


def make_all(normal_train, faulty_test, normal_test, features_train_path, test_onset, out="."):
    out = Path(out)
    (out / "data").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    sensors = [c for c in normal_train.columns if c.startswith("x")]

    # 1. normal operation: average and spread of every variable
    np.savez(out / "models/normal_stats.npz", names=np.array(sensors),
             mean=normal_train[sensors].mean().to_numpy(), std=normal_train[sensors].std().to_numpy())

    # 2. memory index: windows from the first training runs, without the left-out faults
    d = np.load(features_train_path)
    keep = (d["run"] <= INDEX_RUNS) & ~np.isin(d["fault"], EXCLUDED)
    np.savez_compressed(out / "models/memory_index.npz", X=d["X"][keep].astype(np.float32),
                        fault=d["fault"][keep], run=d["run"][keep], start=d["start"][keep])

    # 3. test cases for the app: a few full runs of every state
    runs = pd.concat([faulty_test[faulty_test.simulationRun.isin(DEMO_RUNS)],
                      normal_test[normal_test.simulationRun.isin(DEMO_RUNS)]])
    runs = runs.sort_values(["faultNumber", "simulationRun", "sample"])
    runs["faultNumber"] = runs["faultNumber"].astype(int)
    runs = runs[~runs["faultNumber"].isin(EXCLUDED)]   # faults 3, 9, 15 are not shown in the app demo
    runs["simulationRun"] = runs["simulationRun"].astype(int)
    runs["sample"] = runs["sample"].astype(int)
    runs.round(3).to_csv(out / "data/demo_runs.csv", index=False)

    json.dump({"window": WINDOW, "step": STEP, "test_onset": test_onset, "excluded": EXCLUDED},
              open(out / "models/app_config.json", "w"))
    print("demo_runs.csv:", runs.shape, "| memory index:", int(keep.sum()), "windows")


if __name__ == "__main__":
    import pyreadr
    import sklearn

    def load(name):
        return list(pyreadr.read_r(str(FOLDER / name)).values())[0]

    make_all(load("TEP_FaultFree_Training.RData"), load("TEP_Faulty_Testing.RData"),
             load("TEP_FaultFree_Testing.RData"), FOLDER / "features_train.npz", TEST_ONSET)
    print(f"Done. If you deploy, requirements.txt should contain: scikit-learn=={sklearn.__version__}")
