import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

WINDOW = 20      # samples per window (20 x 3 min = 1 hour)
STEP = 10        # move the window by 10 samples
ONSET = 20       # faults start after sample 20
TRAIN_RUNS = 14  # runs 1-14 = history, runs 15-20 = "new" runs

FAULT_NAMES = {
    0: "Normal operation",
    1: "A/C feed ratio change (step)",
    2: "B composition change (step)",
    4: "Reactor cooling water inlet temperature (step)",
    6: "A feed loss (step)",
    11: "Reactor cooling water inlet temperature (random variation)",
    12: "Condenser cooling water inlet temperature (random variation)",
    13: "Reaction kinetics drift (slow)",
    14: "Reactor cooling water valve sticking",
}

# Placeholder only. The final version will retrieve these from real manuals.
CHECKS = {
    0: "Nothing abnormal found.",
    1: "Check feed flow transmitters and the A and C feed valves.",
    2: "Check the feed composition analyzer and the B feed line.",
    4: "Check the cooling water supply temperature and the reactor temperature loop.",
    6: "Check the A feed flow transmitter, valve and supply pressure.",
    11: "Check the cooling water supply temperature trend and the reactor temperature loop.",
    12: "Check the condenser cooling water supply temperature and condenser valve.",
    13: "Check catalyst activity and the reactor temperature and pressure trends.",
    14: "Check the reactor cooling water valve: positioner, stem friction, controller output vs valve feedback.",
}

VAR_NAMES = {
    "xmeas_7": "Reactor pressure",
    "xmeas_8": "Reactor level",
    "xmeas_9": "Reactor temperature",
    "xmeas_21": "Reactor cooling water outlet temp",
    "xmeas_22": "Separator cooling water outlet temp",
    "xmv_10": "Reactor cooling water flow (valve)",
    "xmv_11": "Condenser cooling water flow (valve)",
}


def nice(name):
    return VAR_NAMES.get(name, name)


@st.cache_resource
def load_everything():
    df = pd.read_csv("data/dataset.csv")
    sensors = [c for c in df.columns if c.startswith("x")]
    t = np.arange(WINDOW)

    features, info = [], []
    for (fault, run), g in df.groupby(["faultNumber", "simulationRun"]):
        values = g.sort_values("sample")[sensors].values
        for start in range(0, len(values) - WINDOW + 1, STEP):
            if fault > 0 and start < ONSET < start + WINDOW:
                continue  # window mixes normal and faulty data, skip it
            w = values[start:start + WINDOW]
            slope = np.polyfit(t, w, 1)[0]
            features.append(np.concatenate([w.mean(0), w.std(0), slope]))
            label = 0 if (fault == 0 or start + WINDOW <= ONSET) else int(fault)
            info.append((label, int(run), start, int(fault)))

    X = np.array(features)
    info = pd.DataFrame(info, columns=["fault", "run", "start", "run_fault"])
    train = (info.run <= TRAIN_RUNS).values

    scaler = StandardScaler().fit(X[train])
    X_scaled = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000).fit(X_scaled[train], info.fault[train])
    memory = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X_scaled[train])

    # normal-operation average and spread, used to see which variables moved
    normal = df[df.faultNumber == 0][sensors]
    return df, sensors, X_scaled, info, train, clf, memory, normal.mean(), normal.std()


df, sensors, X_scaled, info, train, clf, memory, normal_mean, normal_std = load_everything()
history = info[train].reset_index(drop=True)
test = info[~train].reset_index(drop=True)
X_test = X_scaled[~train]

st.title("Process Memory Assistant (Prototype)")
st.write("Pick a 1-hour window from a new Tennessee Eastman run. The app guesses the "
         "fault and finds the most similar past episodes.")

# ---- input ----
cases = test[["run_fault", "run"]].drop_duplicates().sample(frac=1, random_state=0).reset_index(drop=True)
choice = st.selectbox("New run to check", range(len(cases)), format_func=lambda i: f"Test case {i + 1}")
fault, run = cases.run_fault[choice], cases.run[choice]
rows = test.index[(test.run == run) & (test.run_fault == fault)]
starts = sorted(test.start[rows].unique())
start = st.select_slider("Window start (sample number)", options=starts, value=starts[len(starts) // 2])
row = rows[test.start[rows] == start][0]

x = X_test[row].reshape(1, -1)
actual = test.fault[row]

# ---- prediction ----
proba = clf.predict_proba(x)[0]
best = clf.classes_[proba.argmax()]
st.subheader("Prediction")
st.metric("Predicted state", FAULT_NAMES[best], f"{proba.max():.0%} confident")
if st.checkbox("Show the real answer"):
    st.write(f"Real state: **{FAULT_NAMES[actual]}**")
    st.write("Correct" if best == actual else "Wrong")

# ---- memory ----
st.subheader("Most similar past episodes")
dist, idx = memory.kneighbors(x)
similar = history.iloc[idx[0]].copy()
similar["state"] = similar.fault.map(FAULT_NAMES)
similar["similarity"] = (1 - dist[0]).round(3)
st.dataframe(similar[["state", "run", "start", "similarity"]], hide_index=True)

# ---- what moved ----
st.subheader("What moved the most")
window = df[(df.faultNumber == fault) & (df.simulationRun == run)]
window = window.sort_values("sample").iloc[start:start + WINDOW]
z = ((window[sensors].mean() - normal_mean) / normal_std).abs().sort_values(ascending=False)
top = z.head(3).index.tolist()
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a, name in zip(ax, top):
    a.plot(window["sample"], window[name])
    a.axhline(normal_mean[name], color="gray", linestyle="--")
    a.set_title(nice(name), fontsize=9)
    a.set_xlabel("sample")
st.pyplot(fig)
st.caption("Dashed line = average during normal operation.")

# ---- what to check ----
st.subheader("What to check (placeholder)")
st.info(CHECKS[best])
st.caption("These are short hand-written notes for the prototype. The final version will "
           "pull answers, with sources, from real instrument and valve documents.")
