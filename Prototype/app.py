import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from llm_answer import answer
from retrieval import Retriever

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "models" / "app_config.json").read_text())
WINDOW = CONFIG["window"]            # samples per window (20 x 3 min = 1 hour)
STEP = CONFIG["step"]                # the slider moves 10 samples at a time
ONSET = CONFIG["test_onset"]         # sample where the fault starts in the test runs
EXCLUDED = CONFIG["excluded"]        # faults 3, 9, 15: not detectable, left out of the model

VAR_NAMES = {
    "xmeas_1": "A feed flow",
    "xmeas_2": "D feed flow",
    "xmeas_3": "E feed flow",
    "xmeas_4": "A and C feed flow",
    "xmeas_5": "Recycle flow",
    "xmeas_6": "Reactor feed rate",
    "xmeas_7": "Reactor pressure",
    "xmeas_8": "Reactor level",
    "xmeas_9": "Reactor temperature",
    "xmeas_10": "Purge rate",
    "xmeas_11": "Separator temperature",
    "xmeas_12": "Separator level",
    "xmeas_13": "Separator pressure",
    "xmeas_14": "Separator underflow",
    "xmeas_15": "Stripper level",
    "xmeas_16": "Stripper pressure",
    "xmeas_17": "Stripper underflow",
    "xmeas_18": "Stripper temperature",
    "xmeas_19": "Stripper steam flow",
    "xmeas_20": "Compressor work",
    "xmeas_21": "Reactor cooling water outlet temp",
    "xmeas_22": "Separator cooling water outlet temp",
    "xmv_1": "D feed valve",
    "xmv_2": "E feed valve",
    "xmv_3": "A feed valve",
    "xmv_4": "A and C feed valve",
    "xmv_5": "Compressor recycle valve",
    "xmv_6": "Purge valve",
    "xmv_7": "Separator liquid valve",
    "xmv_8": "Stripper liquid valve",
    "xmv_9": "Stripper steam valve",
    "xmv_10": "Reactor cooling water valve",
    "xmv_11": "Condenser cooling water valve",
}
# analyzer readings (mol %)
for i, gas in zip(range(23, 29), "ABCDEF"):
    VAR_NAMES[f"xmeas_{i}"] = f"Reactor feed analysis {gas}"
for i, gas in zip(range(29, 37), "ABCDEFGH"):
    VAR_NAMES[f"xmeas_{i}"] = f"Purge gas analysis {gas}"
for i, gas in zip(range(37, 42), "DEFGH"):
    VAR_NAMES[f"xmeas_{i}"] = f"Product analysis {gas}"


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_data(show_spinner="Writing the answer...")
def cached_answer(situation, sources_tuple, _key):
    sources = [{"title": t, "section": s, "page": p, "text": x} for t, s, p, x in sources_tuple]
    return answer(_key, situation, sources)


def nice(name):
    return VAR_NAMES.get(name, name)


FAULT_TABLE = {row["fault"]: row for row in json.loads((HERE / "rag" / "fault_table.json").read_text())}
FAULT_NAMES = {0: "Normal operation"}
FAULT_NAMES.update({n: row["name"] for n, row in FAULT_TABLE.items()})


def label(fault):
    return FAULT_NAMES[fault] if fault == 0 else f"Fault {fault}: {FAULT_NAMES[fault]}"


@st.cache_resource
def load_retriever():
    """Hybrid (keyword + semantic) search if its files and package are there, else keyword search only."""
    try:
        from rag.hybrid import HybridRetriever
        return HybridRetriever(HERE / "rag" / "chunks.json", HERE / "rag" / "page_embeddings.npz")
    except Exception:
        return Retriever(HERE / "rag" / "chunks.json")


def window_features(w):
    """Mean, standard deviation and slope of every variable in one window (WINDOW x variables)."""
    slope = np.polyfit(np.arange(len(w)), w, 1)[0]
    return np.concatenate([w.mean(0), w.std(0), slope])


@st.cache_resource
def load_everything():
    runs = pd.read_csv(HERE / "data" / "demo_runs.csv")
    sensors = [c for c in runs.columns if c.startswith("x")]
    stats = np.load(HERE / "models" / "normal_stats.npz")
    index = np.load(HERE / "models" / "memory_index.npz")
    info = json.loads((HERE / "models" / "train_info.json").read_text())

    X_index = index["X"]
    history = pd.DataFrame({"fault": index["fault"], "run": index["run"], "start": index["start"]})
    scaler = StandardScaler().fit(X_index)
    try:
        model = joblib.load(HERE / "models" / "gb_model.joblib")
        scaler = joblib.load(HERE / "models" / "scaler.joblib")
    except Exception:
        # saved model does not match this scikit-learn version: retrain on the small memory index
        model = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(X_index, history.fault)
        info["note"] = "retrained on the memory index"
    memory = NearestNeighbors(n_neighbors=5, metric="cosine").fit(scaler.transform(X_index))
    return runs, sensors, stats, history, model, scaler, memory, info


runs, sensors, stats, history, model, scaler, memory, info = load_everything()
normal_mean = pd.Series(stats["mean"], index=stats["names"])
normal_std = pd.Series(stats["std"], index=stats["names"])

st.title("Process Memory Assistant")
st.write("Pick a new Tennessee Eastman run and a 1-hour window. The app guesses the fault, "
         "finds the most similar past episodes, and points to manual pages that may help.")
st.sidebar.markdown(
    "**Model:** gradient boosting on mean, spread and slope of 52 variables.  \n"
    f"**Test macro F1:** {info['test_macro_f1']:.2f} (logistic regression: {info['logreg_test_macro_f1']:.2f}), "
    "on normal operation + 17 faults, on runs the model never saw.  \n"
    f"**Left out:** faults {', '.join(str(f) for f in EXCLUDED)}. With these sensors they can not be told "
    "apart from normal operation.")

# ---- input ----
cases = runs[["faultNumber", "simulationRun"]].drop_duplicates().sample(frac=1, random_state=0).reset_index(drop=True)
show_labels = st.sidebar.checkbox("Show test case labels (gives the answer away)")
choice = st.selectbox("New run to check", range(len(cases)), format_func=lambda i: (
    f"Test case {i + 1}: {label(int(cases.faultNumber[i]))}" if show_labels else f"Test case {i + 1}"))
fault, run = int(cases.faultNumber[choice]), int(cases.simulationRun[choice])
this_run = runs[(runs.faultNumber == fault) & (runs.simulationRun == run)].sort_values("sample").reset_index(drop=True)
starts = list(range(0, len(this_run) - WINDOW + 1, STEP))
start = st.select_slider("Window start (sample number)", options=starts, value=starts[len(starts) // 2])

values = this_run[sensors].to_numpy()
x = window_features(values[start:start + WINDOW]).reshape(1, -1)

# what is really happening in this window
if fault == 0 or start + WINDOW <= ONSET:
    actual = 0
elif start >= ONSET:
    actual = fault
else:
    actual = None    # the fault starts inside this window

# ---- prediction ----
proba = model.predict_proba(x)[0]
best = int(model.classes_[proba.argmax()])
st.subheader("Prediction")
if best != 0:
    st.error(f"ALARM: {label(best)} ({proba.max():.0%} confident)")
else:
    st.success(f"Normal operation ({proba.max():.0%} confident)")
if st.checkbox("Show the real answer"):
    if actual is None:
        st.write(f"Real state: **the fault starts inside this window** ({label(fault)})")
    else:
        st.write(f"Real state: **{label(actual)}**")
        if actual in EXCLUDED:
            st.write("This fault is left out of the model: it can not be told apart from normal operation.")
        else:
            st.write("Correct" if best == actual else "Wrong")

# ---- memory ----
st.subheader("Most similar past episodes")
dist, idx = memory.kneighbors(scaler.transform(x))
similar = history.iloc[idx[0]].copy()
similar["state"] = similar.fault.map(FAULT_NAMES)
similar["similarity"] = (1 - dist[0]).round(3)
st.dataframe(similar[["state", "run", "start", "similarity"]], hide_index=True)

# ---- what deviates most ----
st.subheader("Variables that deviate most from normal")
window = this_run.iloc[start:start + WINDOW]
z_signed = (window[sensors].mean() - normal_mean[sensors]) / normal_std[sensors]
z = z_signed.abs().sort_values(ascending=False)
top = z.head(3).index.tolist()
st.write(f"**{int((z > 3).sum())} of {len(z)} variables** are more than 3 standard deviations away from their normal average in this window.")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, name in zip(ax, top):
    a.plot(this_run["sample"], this_run[name], linewidth=0.8, color="tab:blue")
    a.axvspan(window["sample"].iloc[0], window["sample"].iloc[-1], color="orange", alpha=0.4)
    a.axhline(normal_mean[name], color="gray", linestyle="--")
    if fault != 0:
        a.axvline(ONSET, color="red", linestyle=":")
    a.set_title(f"{nice(name)}\n{z_signed[name]:+.1f} std from normal", fontsize=9)
    a.set_xlabel("sample (1 sample = 3 min)")
handles = [Line2D([0], [0], color="tab:blue", linewidth=1), Patch(color="orange", alpha=0.4),
           Line2D([0], [0], color="gray", linestyle="--"), Line2D([0], [0], color="red", linestyle=":")]
fig.legend(handles, ["sensor reading", "chosen 1-hour window", "normal average", "fault start (real)"],
           loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1))
st.pyplot(fig)
st.caption("These show where the plant moved, not proof of the root cause: control loops spread a fault through many variables, "
           "so a cause and its effects move together.")

# ---- what to check ----
st.subheader("What to check")
user_question = st.text_input("Ask your own question about the manuals (optional)")
moved = " ".join(VAR_NAMES[v] for v in top if v in VAR_NAMES)

query, situation = None, None
if best != 0:
    row = FAULT_TABLE[best]
    st.write(f"**Fault {best}:** {row['name']} ({row['kind'].lower()})")
    st.write("Check first: " + row["check_first"])
    st.caption("Fault description: Downs & Vogel (1993). The 'check first' line is our own note.")
    if row["coverage"] == "none" and not user_question:
        st.warning("The loaded manuals do not cover this fault.")
    elif row["coverage"] == "weak":
        st.warning("The manuals only partly cover this fault, so read the pages with care.")
    if row["coverage"] != "none":
        query = row["search"]   # variable names in the query pulled in unrelated pages, so they only go to the LLM
        situation = (f"Detected fault: {row['name']}. Variables that deviate most from normal (symptoms, not proven causes): {moved}. "
                     "Question: what should the operator check first?")
elif not user_question:
    st.info("Nothing abnormal found.")

if user_question:
    query = user_question
    situation = "Question: " + user_question

if query:
    retriever = load_retriever()
    sources = retriever.search(query, k=5)
    used = getattr(retriever, "last_mode", "keyword")
    st.write("**Manual pages found:**")
    st.caption("Search: keyword + semantic (hybrid)" if used == "hybrid" else "Search: keyword only")
    for i, r in enumerate(sources, start=1):
        where = f"{r['title']}, {r['section']}, page {r['page']}" if r["section"] else f"{r['title']}, page {r['page']}"
        with st.expander(f"[{i}] {where}  (match {r['score']:.2f})"):
            st.write(r["text"])

    key = get_api_key()
    if key:
        if st.button("Write an answer from these pages"):
            sources_tuple = tuple((r["title"], r["section"], r["page"], r["text"]) for r in sources)
            try:
                result = cached_answer(situation, sources_tuple, key)
            except Exception as e:
                st.error("The language model could not be reached. The pages above are still valid.")
                result = None
            if result is not None:
                if not result["points"]:
                    st.warning("The pages above do not answer this, so no answer was written.")
                for p in result["points"]:
                    st.markdown(f"- {p['text']} **[{p['source']}]**")
                for p in result["points"]:
                    src = sources[p["source"] - 1]
                    where = f"{src['title']}, {src['section']}, page {src['page']}" if src["section"] else f"{src['title']}, page {src['page']}"
                    st.markdown(f"> \"{p['quote']}\"  \n> **[{p['source']}] {where}**")
                st.caption(f"Every quote was checked word for word against the page text. "
                           f"{result['dropped']} point(s) removed because the quote could not be verified. "
                           f"Model: {result['model']}.")
    else:
        st.caption("No language model key found, so only the manual pages are shown.")
