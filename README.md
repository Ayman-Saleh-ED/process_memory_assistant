# Process Memory & Instrumentation Troubleshooting Assistant

An assistant that tells a plant operator "have we seen this before?" and "what should I check?" when a process starts to drift.

## Problem Statement
When a process goes abnormal, operators have two problems:
1. Past events are stored as raw sensor data that nobody can search. The "I've seen this before" knowledge lives in the heads of experienced operators.
2. Troubleshooting steps are spread across long instrument and valve manuals, which are slow to search while something is going wrong.

## Impact
- Faster recognition of a developing problem, especially for newer operators and engineers.
- Less time lost searching for the right page in a manual.
- Past experience becomes searchable instead of staying with a few people.
- The assistant is advisory only. It suggests and explains, it never controls the plant.

## Scope
- **Data:** Tennessee Eastman Process, a simulated chemical plant benchmark (extended dataset by Rieth et al., Harvard Dataverse). 52 variables every 3 minutes, normal operation plus 20 faults, 500 simulation runs per state in both the training and the testing file. The prototype used a slice (normal + 8 faults, 20 runs each). The current version uses the full data.
- **Classes (21 states):** normal operation and 20 faults: step changes (1-7), random variations (8-12), slow drift (13), valve sticking (14, 15) and five faults the original description lists as unknown (16-20). The model covers normal operation and 17 of the faults; faults 3, 9 and 15 are left out (see Results). Names are in `Prototype/rag/fault_table.json`.
- **Documents:** 8 public instrument and valve documents (see below).
- **Target audience:** process operators and instrumentation / control engineers.
- **Key deliverables:**
  1. *Process Memory:* compare the current 1-hour window of sensor data with past windows and show the most similar past episodes.
  2. *Troubleshooting assistant:* find the manual pages that match the detected fault, with document name and page number.
  3. A Streamlit app that joins both.
- **Out of scope:** real plant data, closed-loop control, replacing operator judgement.

## Results
**How the model works:** each run is cut into 1-hour windows (20 samples, a new window every 10 samples). For each of the 52 variables a window gets three numbers: mean, standard deviation and slope (156 features). A window is "normal" if it is fault-free or ends before the fault starts; windows that cross the fault start are dropped. A gradient boosting classifier (scikit-learn `HistGradientBoostingClassifier`) predicts the state. Runs 1-400 of the training file train the model, runs 401-500 choose the settings (4 settings compared by macro F1), and the separate testing file (500 runs per state, never used for training or choosing) is used once at the end.

**Faults 3, 9 and 15 are left out.** In the first run with all 20 faults, the model could not tell them from normal operation: half of all normal windows were labelled 3, 9 or 15, and their own recall was 0.26-0.74. With these sensors and these features they are not detectable, so they are reported as a known limit instead of being trained. After training on the other 17, the model calls 98-99% of the windows of faults 3, 9 and 15 "normal".

| Testing file: normal + 17 faults (never-seen runs) | Accuracy | Macro F1 |
|---|---|---|
| Logistic regression (baseline) | 0.942 | 0.946 |
| Gradient boosting (best of 4 settings) | 0.986 | 0.988 |

- Normal windows called normal: 99.3%.
- Weakest states for gradient boosting (recall): faults 10, 13 and 18 (0.95), then 12, 16, 17 and 20 (0.96-0.98). All other states are at 0.99 or above. Logistic regression is clearly worse on faults 10 (0.70) and 16 (0.75).
- Best settings: learning rate 0.05, 63 leaves per tree, 400 rounds (`results/gb_settings.csv`). Per-state results: `results/per_fault_test.csv`. Confusion matrix: `results/confusion_matrix_gb.png`.

**Process Memory** (most similar past windows, cosine similarity on scaled features): the app searches 17,360 windows from 20 training runs per state. On the 9-state prototype slice the closest past window had the right fault 91.4% of the time and the right fault was in the top 5 in 98.3% of cases. This has **not yet been re-measured on the full data**.

**Experiments done for the final submission** (scripts in `Prototype/experiments/`, tables in `Prototype/results/`):

*Detection delay and false alarms* (`detection_delay.py`, on the testing file): 0.69% of normal windows are called a fault. All 17 faults were flagged in 100% of the 500 test runs. The median delay from the fault start to the end of the first window called a fault is 60 minutes for 12 faults (the minimum possible with 1-hour windows), 90 minutes for fault 10, 120 for faults 13, 17 and 20, and 150 for fault 18. The system detects a fault after it has started. It does not predict one in advance: in this data a fault begins as a sudden change and nothing shows in the sensors before it.

*Window length, overlap and features* (`sweep_windows.py`): every row uses the same untuned gradient boosting settings, so rows compare with each other but not with the tuned 0.988 above. The overlap is 50% unless stated.

| Window | Features | Median alarm delay | Test macro F1 |
|---|---|---|---|
| 10 samples (30 min) | basic | 30 min | 0.974 |
| 10 samples | + min, max, last | 30 min | 0.971 |
| **20 samples (1 hour, used)** | **basic** | **60 min** | **0.963** |
| 20 samples | + min, max, last | 60 min | 0.955 |
| 20 samples, step 5 (75% overlap) | basic | 60 min | 0.975 |
| 40 samples (2 hours) | basic | 120 min | 0.991 |
| 40 samples | + min, max, last | 120 min | 0.992 |
| 60 samples (3 hours) | basic | 240 min | 0.993 |
| 60 samples | + min, max, last | 240 min | 0.995 |

Longer windows are more accurate but alarm later; shorter windows alarm faster. One hour is a middle choice, kept because the whole pipeline and the app were built and tested on it. More overlap (more training windows) helped about as much as changing the length. Adding min, max and last value to the summary changed the score by 0.002 or less at 40 and 60 samples and made it slightly worse at 10 and 20, so the three summary numbers are not what limits the model.

*Neural network:* a neural network (MLP, two hidden layers of 128 and 64 units) on the same features and split reached 0.987 macro F1, essentially the same as the tuned gradient boosting (0.988), so it was not adopted. A neural network on the raw windows (CNN or recurrent) was not tried.

**Process Memory on raw data versus features** (`raw_vs_features.py`): not evaluated in this submission. The app uses the summary features.

**Document side.** Each manual page is one chunk, tagged with its section title from the PDF bookmarks (running headers and page numbers removed). Search is **hybrid**: keyword search (TF-IDF) and semantic search (`BAAI/bge-small-en-v1.5` embeddings through the `fastembed` package; pages are embedded in advance and saved in `rag/page_embeddings.npz`) are merged by reciprocal rank fusion. If the embedding package cannot load, the app falls back to keyword search and says so. Evaluated on 24 test questions with known answer pages (the correct pages are defined by a text pattern on each question, independent of the search):

| Search | Right page in top 1 | Top 3 | Top 5 |
|---|---|---|---|
| Keyword | 58% | 79% | 88% |
| Semantic | 63% | 88% | 92% |
| **Hybrid (used)** | **75%** | **92%** | **96%** |

With 24 questions, one question is about 4 points, so small differences are noise. We compared several ways to split the manuals (fixed 1,200 characters, sentences of 80 / 120 / 200 words, one chunk per page; with and without section titles). Adding the section title helped most, and page chunks matched best.

**Written answers.** A language model (Groq, `openai/gpt-oss-120b`) writes the answer from the retrieved pages only. Each point must carry a word-for-word quote, and the app checks that the quote appears in the cited page. Points that fail the check are removed. We evaluated the answers on the same 24 questions plus 6 questions the manuals cannot answer (`rag/evaluate_answers.py`). Every run below used the same model.

| Keyword search, all 30 questions | Result |
|---|---|
| In-scope questions that got an answer | 92% (22 of 24) |
| Points that passed the quote check | 93% (56 of 60) |
| Answered questions whose points cite a correct page | 87% |
| Out-of-scope questions correctly left without an answer | 67% (4 of 6) |

The hybrid run is only partly comparable. Its first 22 in-scope questions used the same model and gave an answer for 21 of them (quote check: 58 of 61 points passed). After that the free plan's rate limit pushed the last 7 questions onto a smaller fallback model that returned no points at all, so the hybrid figures for those, and the out-of-scope result, are not reliable. The "correct page" share for hybrid (80%) is also pessimistic, because the correct pages are defined by a text pattern that favours keyword matches.

The quote check proves that a quote exists on the page, not that it answers the question. For two out-of-scope questions (heat exchanger repair and gas chromatograph choice) the keyword run returned points with real quotes from unrelated pages. The quote is shown under every point so the reader can judge.

**Measurables used to compare the baseline and the model**

| Measure | Logistic regression | Gradient boosting |
|---|---|---|
| Accuracy (normal + 17 faults, unseen runs) | 0.942 | 0.986 |
| Macro F1 | 0.946 | 0.988 |
| Normal windows called normal (false alarms) | 94.2% | 99.3% |
| Faults flagged in the test runs | not measured | 100% (17 of 17 faults) |
| Median alarm delay (1-hour windows) | not measured | 60 min (best possible) |

For the manual search the measures are the page hit rate (top 1, 3, 5), the share of points that pass the quote check, and the share of out-of-scope questions left unanswered.

Known limits:
- The instrument manuals cover valve and sensor problems well, but cooling water supply, feed composition and reaction chemistry only weakly or not at all. Most of these faults are process faults, not instrument faults, so more instrument manuals would not close that gap. The app shows a coverage rating per fault (`rag/fault_table.json`) and says when the manuals do not cover a fault.
- Faults 3, 9 and 15 (D feed temperature, random D feed temperature, condenser cooling water valve sticking) cannot be detected from these sensors with these features, so the model does not cover them.
- Faults 16 to 20 are unknown in the original description, so their causes are not stated.
- The variables shown as "deviating most from normal" are symptoms, not proven causes: control loops spread a fault through many variables.
- The "check first" lines in `rag/fault_table.json` are our own notes from the fault descriptions, not taken from a source and not reviewed by a plant expert. Fault names and types come from Downs & Vogel (1993). The coverage ratings are a judgement made after trying searches on the manuals.
- Scores on the data side are high because the data is simulated. Real plant data is noisier.
- The model only knows the faults it was trained on. A new kind of fault will be labelled as the closest known one.
- The system detects a fault after it starts (at best one window later). It does not forecast one. We did not measure how fast a human operator notices these faults, so we do not claim the system is faster. The intended benefit is different: it watches all 52 variables at once, names the most likely fault, shows similar past episodes and points to the manual pages, which is the part that takes an operator time. A fair comparison with operators or with the plant's own alarms would be a next step.
- The answer evaluation uses 24 in-scope and 6 out-of-scope questions, which is small.

## Documents used
Not included in this repository (copyrighted). Download them into `Prototype/docs/` to rebuild the index.
- [Emerson Control Valve Handbook](https://www.emerson.com/en-us/automation/valves/controlvalves/control-valve-handbook)
- [Fisher FIELDVUE DVC6200 manual](https://emerson.com/documents/automation/instruction-manual-fieldvue-dvc6200-hw2-digital-valve-controller-en-en-11382598.pdf)
- [Siemens SIPART PS2 HART operating instructions](https://cache.industry.siemens.com/dl/files/419/109782419/att_1034214/v1/A5E00074631-AFen_PS2HART_OI_en-US.pdf)
- [Rosemount 3051 reference manual](https://emerson.com/documents/automation/reference-manual-rosemount-3051-pressure-transmitter-en-76002.pdf)
- [Rosemount 3144P temperature transmitter](https://core.instrumart.com/assets/Rosemount-3144P-Manual.pdf)
- [Rosemount 8700M magnetic flowmeter](https://core.instrumart.com/assets/Rosemount-8700M-manual.pdf)
- [ISA: closed-loop control troubleshooting](https://www.isa.org/intech-home/2018/march-april/departments/closed-loop-control-troubleshooting)
- [AIChE CEP: control loop troubleshooting](https://www.aiche.org/sites/default/files/cep/20210638r.pdf)

## Repository structure
```
Prototype/
  app.py                      Streamlit app
  prepare_full_data.py        RData files -> window features (full_data/*.npz)
  train_models.py             trains logistic regression + gradient boosting, writes models/ and results/
  make_demo_files.py          small files the deployed app needs (demo runs, memory index, normal stats)
  retrieval.py                keyword search over the manual pages
  llm_answer.py               written answer with quote checking (retries on rate limits)
  experiments/                window / feature sweep, detection delay, raw-versus-features test
  build_index.py              PDFs -> rag/chunks.json (one chunk per page)
  models/                     gb_model.joblib, scaler.joblib, memory_index.npz, normal_stats.npz, train_info.json, app_config.json
  results/                    per-state results, settings comparison, confusion matrix, window sweep, detection delay, search comparison
  data/demo_runs.csv          test runs shown in the app (runs 1-2 of each state, testing file)
  data/dataset.csv            slice of the data used by the prototype notebook
  notebooks/prototype.ipynb   prototype baselines and evaluation
  rag/
    chunks.json               page text with document, section and page
    fault_table.json          the 20 faults, what to check first, coverage
    questions.py              24 test questions + 6 out-of-scope questions
    hybrid.py                 hybrid (keyword + semantic) search; builds page_embeddings.npz
    page_embeddings.npz       page embeddings made in advance
    evaluate.py               keyword search evaluation
    evaluate_hybrid.py        keyword vs semantic vs hybrid
    evaluate_answers.py       checks the written answers (quote check, out-of-scope questions)
    check_groq.py             lists the Groq models the key can use
    evaluate_embeddings.py    earlier embedding comparison
  requirements.txt
  .gitignore
```
`full_data/` (the large downloaded and processed data) and `docs/` (the manuals) are not in the repository.

## How to run
**Just the app** (all needed files are in the repository):
```
cd Prototype
pip install -r requirements.txt
streamlit run app.py
```
For the written answer, add a free Groq key in `Prototype/.streamlit/secrets.toml`: `GROQ_API_KEY = "your-key"`. Without a key the app shows the manual pages only.

**Rebuild the model from the data:**
1. Download the four RData files from [Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1) (`TEP_FaultFree_Training`, `TEP_Faulty_Training`, `TEP_FaultFree_Testing`, `TEP_Faulty_Testing`) into `Prototype/full_data/`.
2. `pip install pyreadr`
3. `python prepare_full_data.py` (windows and features)
4. `python train_models.py` (about 10 minutes)
5. `python make_demo_files.py`

**Experiments** (need `full_data/`): `python experiments/detection_delay.py`, `python experiments/sweep_windows.py mlp` (30-40 minutes), `python experiments/raw_vs_features.py`.

**Search and answer evaluation:** `pip install fastembed`, `python rag/hybrid.py` (builds the page embeddings), `python rag/evaluate_hybrid.py`; with a Groq key in the environment, `python rag/evaluate_answers.py` (add `hybrid` for the hybrid search; about 10 minutes because of rate limits).

**Rebuild the manual index:** put the PDFs in `Prototype/docs/`, then run `python build_index.py` and `python rag/evaluate.py`.

## Next steps
- CNN or recurrent networks on the raw windows (the MLP on summary features tied with gradient boosting), and an anomaly detector trained on normal data only, to flag unfamiliar behaviour.
- Tune gradient boosting together with window length and overlap, using a cross-validated search grouped by run.
- Measure Process Memory on raw windows versus summary features, and on the full data.
- Repeat the hybrid answer evaluation on a plan without rate limits, and add more questions.

## Acknowledgments
Data: Rieth, C. A., Amsel, B. D., Tran, R., Cook, M. B. (2017), *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, Harvard Dataverse. Original process and fault list: Downs & Vogel (1993). Manuals belong to their manufacturers.
