"""How fast does the model flag a fault, and how often does it raise a false alarm?
Uses the predictions saved by train_models.py. Takes a few seconds, needs no RData files.

Run from the Prototype folder:
    python experiments/detection_delay.py
Saves results/detection_delay.csv
"""
import numpy as np
import pandas as pd

WINDOW, TEST_ONSET, MINUTES = 20, 160, 3
EXCLUDED = [3, 9, 15]

test = np.load("full_data/features_test.npz")
keep = ~np.isin(test["fault"], EXCLUDED)                      # same filter as train_models.py
pred = np.load("results/test_predictions.npz")["gb"]
df = pd.DataFrame({"label": test["fault"][keep], "run": test["run"][keep], "start": test["start"][keep], "pred": pred})
assert len(df) == len(pred), "predictions and windows do not line up: run train_models.py again"

normal = df[df.label == 0]
print(f"False alarms: {(normal.pred != 0).mean():.2%} of normal windows are called a fault")

rows = []
for fault, g in df[df.label > 0].groupby("label"):
    runs = g.run.nunique()
    any_flag = g[g.pred != 0].groupby("run").start.min()
    right = g[g.pred == fault].groupby("run").start.min()
    rows.append(dict(
        fault=int(fault), runs=runs,
        flagged_share=round(len(any_flag) / runs, 3),
        median_delay_to_flag_min=float(((any_flag + WINDOW - TEST_ONSET) * MINUTES).median()),
        named_right_share=round(len(right) / runs, 3),
        median_delay_to_right_name_min=float(((right + WINDOW - TEST_ONSET) * MINUTES).median()),
    ))
table = pd.DataFrame(rows)
print(table.to_string(index=False))
print(f"\nAll faults: median delay to flag {table.median_delay_to_flag_min.median():.0f} min, "
      f"to the right name {table.median_delay_to_right_name_min.median():.0f} min")
print("Delay = minutes from the fault start to the END of the first window that is called a fault.")
print("With 1-hour windows the minimum possible is 60 min, because the first window that is fully after the start ends 1 hour later.")
table.to_csv("results/detection_delay.csv", index=False)
