<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=200&section=header&text=Fraud%20Detection%20Engine&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-time%20credit%20card%20fraud%20scoring&descAlignY=55&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.977-success?style=for-the-badge)](#results)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=18&pause=1000&color=2A5298&center=true&vCenter=true&width=600&lines=Classifies+284%2C807+real+transactions;Only+0.17%25+of+them+are+fraud;Logistic+Regression+vs+Random+Forest;Live+in-browser+risk+scoring+dashboard)](https://git.io/typing-svg)

</div>

---

## 📖 Overview

A fraud-detection pipeline trained on **284,807 real anonymized transactions**, only **492 (0.17%) of which are fraudulent** — a severe class-imbalance problem tackled head-on rather than papered over with accuracy. The project ships two things: a reproducible **training pipeline** and a **zero-install, single-file browser dashboard** that scores transactions live using the real trained model.

---

## 🏗️ Architecture

<div align="center">
<img src="01-fraud-architecture.svg" alt="Fraud detection animated architecture diagram" width="100%"/>
</div>

<details>
<summary>Mermaid source (fallback / editable)</summary>

```mermaid
flowchart LR
    A[("creditcard.csv<br/>284,807 rows")] --> B[EDA & Feature Ranking]
    B --> C{Imbalance Strategy}
    C -->|"class_weight='balanced'"| D[Logistic Regression]
    C -->|3:1 undersampling| E[Random Forest]
    D --> F[Model Evaluation]
    E --> F
    F --> G[["COEF / INTERCEPT<br/>exported to JS"]]
    G --> H["fraud_detection_console.html<br/>(client-side scoring)"]

    style A fill:#1e3c72,color:#fff
    style H fill:#2a5298,color:#fff
    style F fill:#0f6b3a,color:#fff
```

</details>

> The Logistic Regression model is the one deployed to the dashboard — its coefficients are simple enough to embed directly in JavaScript, coming within a hair of the Random Forest's accuracy without needing a server.

---

## 📁 Project Structure

```text
.
├── fraud_detection_model.py       # EDA, feature selection, model training, evaluation, predict_fraud()
├── fraud_detection_console.html   # standalone interactive dashboard (open directly, no server)
└── creditcard_sample.csv          # ~600-row excerpt for column reference
```

> The full dataset (150 MB, 284,807 rows) isn't bundled. To retrain, drop your own `creditcard.csv` (columns: `Time, V1..V28, Amount, Class`) into this folder.

---

## 🔬 Methodology

### Feature selection
Every one of the 30 features was ranked by how far its fraud-class mean sits from its genuine-class mean (in standard deviations), cross-checked against Random Forest importance scores. **10 features carried real signal:**

`V14 · V4 · V10 · V12 · V17 · V11 · V3 · V16 · V7 · Amount`

`Time` and 18 of the 28 anonymized PCA components were dropped for barely separating fraud from genuine transactions.

### Handling severe imbalance
At 0.17% fraud, a naive classifier hits 99.8% accuracy by predicting "genuine" every time — worthless. Two strategies were compared head-to-head:

| Strategy | Outcome |
|---|---|
| `class_weight='balanced'` on full data | ✅ **Winner** — best precision/recall trade-off |
| Manual 3:1 undersampling | Discarded too much genuine data; precision dropped sharply |

---

## 📊 Results

Evaluated on **56,962 held-out transactions** the model never saw during training:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression (tuned threshold) | 0.82 | 0.82 | 0.82 | 0.971 |
| **Random Forest** | **0.84** | 0.82 | **0.83** | **0.977** |

Random Forest edges ahead, but a 300-tree forest can't live inside a static HTML page — so the dashboard runs **Logistic Regression** client-side instead.

---

## ⚡ Quick Start

### Train / retrain the model
```bash
pip install pandas numpy scikit-learn
python fraud_detection_model.py
```
This reprints the EDA, feature ranking, imbalance comparison, and final metrics. Copy the refreshed `COEF` / `INTERCEPT` / `Z_THRESHOLD` values into `fraud_detection_console.html`'s `<script>` block to update the dashboard.

### Run the dashboard
Just double-click `fraud_detection_console.html` — everything (sample data, model coefficients, charting) is bundled in one file. Load a sample transaction or set your own values across the 10 sliders, then hit **Scan transaction**.

---

## 🖥️ Reading the Dashboard

| Element | What it shows |
|---|---|
| **Verdict badge** | Fraud / Genuine, from the model's decision boundary |
| **Fraud risk score (0–100)** | Normalized model score; the gauge's line marks the decision boundary |
| **Top signals** | Which of the 10 features pushed the verdict toward fraud (red) or genuine (green) |
| **Scatter plot** | This transaction plotted against 400 real transactions on `V14` vs `V4` — fraud clusters tightly lower-right |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2a5298,100:1e3c72&height=100&section=footer" width="100%"/>
</div>
