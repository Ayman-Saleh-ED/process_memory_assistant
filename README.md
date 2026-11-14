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
- **Data:** Tennessee Eastman Process, a simulated chemical plant benchmark (extended dataset by Rieth et al., Harvard Dataverse). A slice is used here: normal operation plus 8 faults (1, 2, 4, 6, 11, 12, 13, 14), 20 runs each.
- **Target audience:** process operators and instrumentation / control engineers.
- **Key deliverables:**
  1. *Process Memory:* compare the current 1-hour window of sensor data with past windows and show the most similar past episodes.
  2. *Troubleshooting assistant:* retrieve answers, with sources, from public instrument and valve documents (RAG).
  3. A Streamlit app that joins both.
- **Out of scope:** real plant data, closed-loop control, replacing operator judgement.

## Prototype status
The prototype lives in `Prototype/`.
- **Done:** data pipeline, windowing and features, two baselines, Streamlit app.
- **Placeholder:** the "what to check" box uses short hand-written notes. The document search (RAG) is the next step.

| Baseline (unseen runs 15-20) | Result |
|---|---|
| Always guess "normal" | 13% |
| Logistic regression, fault type | 97.7% accuracy, 0.978 macro F1 |
| Process Memory: closest past window has the right fault | 91.4% |
| Process Memory: right fault is in the top 5 | 98.3% |

Scores are high because the data is *cleann* and the faults are clear.

## Repository structure
```
Prototype/
  notebooks/prototype.ipynb   baseline model and evaluation
  data/dataset.csv            Tennessee Eastman slice (90,000 rows)
  app.py                      Streamlit app
  requirements.txt
  .gitignore
```

## How to run
```
cd Prototype
pip install -r requirements.txt
streamlit run app.py
```
The notebook is in `Prototype/notebooks/` and reads `../data/dataset.csv`.

## Next steps
- Train a neural network to create the embeddings and compare with the baselines.
- Test on faults the model has never seen, with a "nothing similar found" threshold.
- Build the RAG part over public instrument and valve documents.
- Evaluate retrieval with a set of test questions, and check that answers cite sources.

## Acknowledgments
Data: Rieth, C. A., Amsel, B. D., Tran, R., Cook, M. B. (2017), *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, Harvard Dataverse. Original process: Downs & Vogel (1993).
