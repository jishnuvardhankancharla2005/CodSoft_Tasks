<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:11998e,100:38ef7d&height=200&section=header&text=Sales%20Prediction%20Lab&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Advertising%20spend%20%E2%86%92%20Sales%20forecast&descAlignY=55&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![R²](https://img.shields.io/badge/R²-0.954-success?style=for-the-badge)](#-results)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=18&pause=1000&color=11998E&center=true&vCenter=true&width=600&lines=200+advertising+campaigns%2C+3+channels;Polynomial+regression+on+TV+%2B+Radio;Newspaper+dropped+%E2%80%94+almost+zero+signal;Live+in-browser+forecasting+dashboard)](https://git.io/typing-svg)

</div>

---

## 📖 Overview

Predicts advertising **Sales** from **TV**, **Radio**, and **Newspaper** spend across 200 historical campaigns — and ships a single-file interactive dashboard that runs the real trained model directly in the browser.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[("advertising.csv<br/>200 campaigns")] --> B[EDA & Correlation Analysis]
    B --> C[Feature Selection]
    C -->|dropped| D["Newspaper<br/>(corr 0.16, importance 1.3%)"]
    C -->|kept| E["TV + Radio<br/>(corr 0.90, importance 85%)"]
    E --> F{Model Comparison}
    F --> G[Linear Regression]
    F --> H[["Polynomial Regression<br/>(degree 2) — deployed"]]
    F --> I[Random Forest]
    H --> J["COEF exported to JS"]
    J --> K["sales_prediction_lab.html<br/>(client-side forecast)"]

    style A fill:#11998e,color:#fff
    style H fill:#0f6b3a,color:#fff
    style K fill:#38ef7d,color:#000
```

---

## 📁 Project Structure

```text
.
├── advertising.csv               # source dataset: TV, Radio, Newspaper spend + Sales
├── sales_prediction_model.py     # EDA, feature selection, model comparison, predict_sales()
└── sales_prediction_lab.html     # standalone interactive dashboard (open directly, no server)
```

---

## 🔬 Methodology

**Polynomial regression (degree 2)** on **TV** and **Radio** spend only. `Newspaper` was dropped after feature selection showed it contributes almost nothing — **0.16 correlation**, **1.3% Random Forest importance** — against TV's **0.90 correlation** and **85% importance**. The polynomial terms capture the real-world synergy between TV and radio campaigns running together.

---

## 📊 Results

Performance on held-out test data:

| Metric | Value |
|---|:---:|
| R² | **0.954** |
| RMSE | 1.19 (thousand units) |
| MAE | 0.88 (thousand units) |
| Training samples | 200 (80/20 split) |

---

## ⚡ Quick Start

### Train / retrain the model
```bash
pip install pandas numpy scikit-learn
python sales_prediction_model.py
```
Reprints the EDA, feature importances, model comparison table, and final coefficients — copy the refreshed `COEF` values into `sales_prediction_lab.html`'s `<script>` block to update the dashboard.

### Run the dashboard
Just double-click `sales_prediction_lab.html`, or open it in a browser. Everything (data, coefficients, charting) is bundled in the single file — nothing else to install. Move the **TV** and **Radio** faders, click **Run forecast**, and it predicts Sales and plots the result against all 200 historical campaigns.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:38ef7d,100:11998e&height=100&section=footer" width="100%"/>
</div>
