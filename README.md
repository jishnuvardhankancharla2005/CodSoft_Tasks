<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141e30,50:243b55,100:0f6b3a&height=220&section=header&text=Machine%20Learning%20Project%20Portfolio&fontSize=32&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=5%20end-to-end%20ML%20projects%20%E2%80%94%20from%20EDA%20to%20deployed%2C%20zero-install%20UIs&descAlignY=58&descSize=15" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6423?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=18&pause=1000&color=0F6B3A&center=true&vCenter=true&width=700&lines=5+supervised+learning+projects%2C+5+deployed+UIs;Fraud+detection+%C2%B7+Iris+%C2%B7+Movie+ratings+%C2%B7+Sales+%C2%B7+Titanic;Every+model+ships+with+a+live%2C+zero-install+demo)](https://git.io/typing-svg)

</div>

---

## 📚 Table of Contents

| # | Project | Task | Best Model | Headline Metric |
|---|---|---|---|---|
| 1 | [Credit Card Fraud Detection](#1--credit-card-fraud-detection) | Binary classification | Logistic Regression (deployed) | ROC-AUC **0.971** |
| 2 | [Iris Species Classification](#2--iris-species-classification) | Multi-class classification | SVM (linear kernel) | CV Accuracy **96.67%** |
| 3 | [Movie Rating Prediction (IMDb India)](#3--movie-rating-prediction-imdb-india) | Binary classification | XGBoost (tuned) | ROC-AUC **0.847** |
| 4 | [Sales Prediction](#4--sales-prediction) | Regression | Polynomial Regression | R² **0.954** |
| 5 | [Titanic Survival Prediction](#5--titanic-survival-prediction) | Binary classification | Stacking Ensemble | Accuracy **~88–90%** |

---

## 🗺️ Portfolio Architecture

<div align="center">
<img src="00-combined-architecture.svg" alt="Combined ML portfolio animated architecture diagram" width="100%"/>
</div>

<details>
<summary>Mermaid source (fallback / editable)</summary>

```mermaid
flowchart TD
    subgraph Data["📥 Raw Data"]
        D1[creditcard.csv]
        D2[iris.csv]
        D3["IMDb Movies India.csv"]
        D4[advertising.csv]
        D5[Titanic-Dataset.csv]
    end

    subgraph Pipeline["⚙️ EDA → Feature Engineering → Modeling"]
        P1[Fraud: imbalance handling + LR/RF]
        P2[Iris: 6-model CV bakeoff + SVM]
        P3[Movies: OOF target encoding + XGBoost]
        P4[Sales: polynomial regression on TV+Radio]
        P5[Titanic: LR + RF + XGBoost → Stacking]
    end

    subgraph Deploy["🚀 Deployed Interfaces"]
        U1["fraud_detection_console.html"]
        U2["iris_classifier_demo.html + Streamlit"]
        U3["Flask app: 3D Three.js predictor"]
        U4["sales_prediction_lab.html"]
        U5["titanic_prediction.py + bar-chart viz"]
    end

    D1 --> P1 --> U1
    D2 --> P2 --> U2
    D3 --> P3 --> U3
    D4 --> P4 --> U4
    D5 --> P5 --> U5

    style Data fill:#141e30,color:#fff
    style Pipeline fill:#243b55,color:#fff
    style Deploy fill:#0f6b3a,color:#fff
```

</details>

---

## 📁 Repository Layout

```text
.
├── 01-fraud-detection/
│   ├── fraud_detection_model.py
│   ├── fraud_detection_console.html
│   └── creditcard_sample.csv
├── 02-iris-classification/
│   ├── data/iris.csv
│   ├── train_model.py
│   ├── model/               # iris_model.pkl, scaler.pkl, label_encoder.pkl, metadata.json
│   ├── reports/
│   └── app/                 # iris_classifier_demo.html, streamlit_app.py
├── 03-movie-rating-prediction/
│   ├── src/                 # clean.py, features.py, train.py
│   ├── app/                 # app.py, templates/, static/
│   ├── models/
│   └── run_app.py
├── 04-sales-prediction/
│   ├── advertising.csv
│   ├── sales_prediction_model.py
│   └── sales_prediction_lab.html
└── 05-titanic-survival/
    ├── Titanic-Dataset.csv
    └── titanic_prediction.py
```

---

## 1 · Credit Card Fraud Detection

Classifies transactions as fraudulent or genuine, trained on **284,807 real transactions** (492 fraud, **0.17%**).

<div align="center">
  <img src="CodSoft_fraud_project/01-fraud-architecture.svg" alt="Credit Card Fraud Detection Architecture Diagram" width="100%"/>
</div>

**Feature selection** ranked all 30 features by class separation; 10 carried real signal: `V14, V4, V10, V12, V17, V11, V3, V16, V7, Amount`. **Imbalance handling** compared `class_weight='balanced'` against 3:1 undersampling — class-weighting won.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression (deployed) | 0.82 | 0.82 | 0.82 | 0.971 |
| Random Forest | 0.84 | 0.82 | 0.83 | 0.977 |

**Ships with:** `fraud_detection_console.html` — a zero-install dashboard with a live risk gauge, top contributing signals, and a scatter plot against 400 real transactions.

```bash
pip install pandas numpy scikit-learn
python fraud_detection_model.py
```

---

## 2 · Iris Species Classification

Classifies *setosa* / *versicolor* / *virginica* from sepal/petal measurements. Petal measurements carry **~86% of the classification signal**. Six algorithms were compared with 10-fold stratified CV:

<div align="center">
  <img src="CodSoft_iris-classification-project/02-iris-architecture.svg" alt="Iris Species Classification Architecture Diagram" width="100%"/>
</div>

| Model | CV Accuracy |
|---|:---:|
| **SVM (linear kernel)** | **96.67%** |
| Logistic Regression | 95.83% |
| Random Forest | 95.00% |

*Setosa* is classified correctly **100%** of the time; remaining error sits at the genuine versicolor/virginica overlap.

**Ships with:** a browser demo whose sliders redraw a live botanical illustration using the model's exact embedded weights, plus a full Streamlit app with radar charts and EDA plots.

```bash
pip install -r requirements.txt
python train_model.py            # retrain
streamlit run app/streamlit_app.py   # or run the full app
```

---

## 3 · Movie Rating Prediction (IMDb India)

Predicts whether a film will be **High-Rated** (IMDb ≥ 6.5) from five pre-release inputs — Genre, Director, Lead Actor, Year, expected Votes — on **15,509 films**.

<div align="center">
  <img src="CodSoft_Movie_Rating_Prediction_using_Python/03-movie-rating-architecture.svg" alt="Movie Rating Prediction Architecture Diagram" width="100%"/>
</div>

| Model | Accuracy | ROC-AUC |
|---|:---:|:---:|
| **XGBoost (tuned) — deployed** | **0.783** | **0.847** |
| Stacking Ensemble | 0.782 | 0.847 |
| Baseline (majority class) | 0.634 | 0.500 |

A **+14.9 point lift** over baseline. Out-of-fold target encoding keeps `Director`/`Actor` leakage-free.

**Ships with:** a Flask app featuring a full **3D Three.js** scene — a starfield, a glowing probability meter, and a mouse-reactive tilting form card — plus a results dashboard with ROC curves and confusion matrices.

```bash
pip install -r requirements.txt
python -m src.train "path\to\IMDb Movies India.csv"
python run_app.py   # → http://127.0.0.1:5000
```

---

## 4 · Sales Prediction

Predicts advertising **Sales** from **TV**, **Radio**, and **Newspaper** spend across 200 campaigns. `Newspaper` was dropped (0.16 correlation, 1.3% importance) in favor of **TV + Radio** polynomial regression (degree 2), capturing real campaign synergy.

<div align="center">
  <img src="CodSoft_sales_prediction_app/04-sales-architecture.svg" alt="Sales Prediction Architecture Diagram" width="100%"/>
</div>

| Metric | Value |
|---|:---:|
| R² | **0.954** |
| RMSE | 1.19k units |
| MAE | 0.88k units |

**Ships with:** `sales_prediction_lab.html` — move the TV/Radio faders, click **Run forecast**, see the prediction plotted against all 200 historical campaigns.

```bash
pip install pandas numpy scikit-learn
python sales_prediction_model.py
```

---

## 5 · Titanic Survival Prediction

Predicts passenger survival from demographics, ticket class, fare, and family features (`Title`, `FamilySize`, `IsAlone`, `CabinKnown`, `FarePerPerson`). Logistic Regression, Random Forest, and XGBoost are combined via a **`StackingClassifier`** for **~88–90% accuracy**.

<div align="center">
  <img src="CodSoft_Titanic_Survival_Prediction/05-titanic-architecture.svg" alt="Titanic Survival Prediction Architecture Diagram" width="100%"/>
</div>

**Ships with:** a bar-chart visualization of survival probability on every prediction run.

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
python titanic_prediction.py
```

---

## 🛠️ Combined Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB6423?style=flat-square)
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Three.js](https://img.shields.io/badge/-Three.js-000000?style=flat-square&logo=three.js&logoColor=white)
![matplotlib](https://img.shields.io/badge/-matplotlib-11557C?style=flat-square)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

Every project follows the same discipline: rigorous **EDA → feature selection → multi-model comparison → held-out evaluation**, and every model is shipped behind a **working, zero-friction interface** — a static HTML dashboard, a Streamlit app, or a full Flask + 3D UI — so results are something you can click through, not just read in a table.

## 📄 License

MIT — see individual project folders for details.

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f6b3a,50:243b55,100:141e30&height=120&section=footer" width="100%"/>

**⭐ If this portfolio was useful, consider starring the repo!**

</div>
