---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# Smart MCQ Solver

A BiLSTM-based ranking model that scores multiple-choice options and returns
the top-3 most likely answers, deployed with Streamlit.

## How it works

Each option is paired with the question (`prompt [SEP] option`) and scored
independently through the model as a binary "is this option correct"
classification. Options are then ranked by predicted probability, and the
top 3 are shown.

## Files

- `app.py` — Streamlit UI
- `inference.py` — loads the model/tokenizer/config and exposes `predict()`
- `model.py` — the `BiLSTMClassifier` architecture
- `models/` — trained artifacts (`model.pth`, `tokenizer.pkl`, `config.json`)
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
