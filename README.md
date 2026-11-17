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

**Document side (first version):** each manual page is one chunk, tagged with its section title from the PDF bookmarks (running headers and page numbers removed). Pages are found with keyword search (TF-IDF). Evaluated on 24 test questions with known answer pages:

| Keyword search | Result |
|---|---|
| Right page in top 1 | 58% |
| Right page in top 3 | 79% |
| Right page in top 5 | 88% |

We compared several ways to split the manuals (fixed 1,200 characters, sentences of 80 / 120 / 200 words, one chunk per page; with and without section titles). Adding the section title helped most. Page chunks matched best and start at the beginning of a sentence 95% of the time, against 46% for fixed-size cuts. On 24 questions, differences of one or two questions are noise.

The answer is written by a language model (Groq) from the retrieved pages only. Each point must carry a word-for-word quote, and the app checks that the quote appears in the cited page. Points that fail the check are removed. The check proves that the quote exists on the page, not that the one-sentence summary is a perfect reading of it, so the quote is shown under every point.

Known limits:
- The search scores for questions the manuals cannot answer overlap with scores for real ones, so a score threshold cannot decide "not covered". The app uses a per-fault coverage rating instead (`rag/fault_table.json`).
- The instrument manuals cover valve and sensor problems well, but cooling water supply, feed composition and reaction chemistry only weakly or not at all. The app says so.
- Faults 3, 9 and 15 (D feed temperature, random D feed temperature, condenser cooling water valve sticking) cannot be detected from these sensors with these features, so the model does not cover them.
- Faults 16 to 20 are unknown in the original description, so their causes are not stated.
- The "check first" lines in `rag/fault_table.json` are our own notes from the fault descriptions, not taken from a source and not reviewed by a plant expert. Fault names and types come from Downs & Vogel (1993). The coverage ratings are a judgement made after trying searches on the manuals.
- Scores on the data side are high because the data is simulated. Real plant data is noisier.
- The model only knows the faults it was trained on. A new kind of fault will be labelled as the closest known one (the planned anomaly detector addresses this).

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
  retrieval.py                manual search
  llm_answer.py               written answer with quote checking
  build_index.py              PDFs -> rag/chunks.json (one chunk per page)
  models/                     gb_model.joblib, scaler.joblib, memory_index.npz, normal_stats.npz, train_info.json, app_config.json
  results/                    per-state results, settings comparison, confusion matrix, test predictions
  data/demo_runs.csv          test runs shown in the app (runs 1-2 of each state, testing file)
  data/dataset.csv            slice of the data used by the prototype notebook
  notebooks/prototype.ipynb   prototype baselines and evaluation
  rag/
    chunks.json               page text with document, section and page
    fault_table.json          the 20 faults, what to check first, coverage
    questions.py              24 test questions + 6 out-of-scope questions
    evaluate.py               retrieval evaluation
    evaluate_embeddings.py    optional embedding comparison
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

**Rebuild the manual index:** put the PDFs in `Prototype/docs/`, then run `python build_index.py` and `python rag/evaluate.py`.

## Next steps
- Neural networks (GRU / 1D-CNN) on the raw windows, so the model learns its own features. Their embeddings will be used for Process Memory and compared with the current version.
- Anomaly detector trained on normal data only, to flag unknown faults and measure detection delay and false alarms.
- Measure Process Memory on the full data.
- Embedding-based manual search, compared with the keyword search.
- Tune gradient boosting with a grouped (by run) cross-validated grid search.

## Acknowledgments
Data: Rieth, C. A., Amsel, B. D., Tran, R., Cook, M. B. (2017), *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, Harvard Dataverse. Original process and fault list: Downs & Vogel (1993). Manuals belong to their manufacturers.
