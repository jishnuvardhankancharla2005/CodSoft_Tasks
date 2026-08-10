<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a2980,100:26d0ce&height=200&section=header&text=Titanic%20Survival%20Prediction&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Who%20made%20it%20off%20the%20lifeboats%3F&descAlignY=55&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6423?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-~88--90%25-success?style=for-the-badge)](#-model-training)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=18&pause=1000&color=1A2980&center=true&vCenter=true&width=600&lines=Classic+Titanic+passenger+dataset;Logistic+Regression+%2B+Random+Forest+%2B+XGBoost;StackingClassifier+ensemble+for+max+accuracy;Bar-chart+survival+probability+on+every+run)](https://git.io/typing-svg)

</div>

---

## 📖 Project Overview

Predicts whether a passenger aboard the Titanic survived, using the classic Titanic dataset — passenger age, sex, ticket class, fare, family size, and embarkation port. The project demonstrates a full, professional end-to-end pipeline:

- Data cleaning and preprocessing
- Feature engineering (titles, family size, cabin presence, fare normalization)
- Model training with **Logistic Regression**, **Random Forest**, and **XGBoost**
- **Ensemble learning** (`StackingClassifier`) for optimized accuracy
- Visualization of survival probabilities on every prediction

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[("Titanic-Dataset.csv")] --> B[Data Cleaning<br/>missing values, drop irrelevant cols]
    B --> C["Feature Engineering<br/>Title, FamilySize, IsAlone,<br/>CabinKnown, FarePerPerson"]
    C --> D{Model Training}
    D --> E[Logistic Regression]
    D --> F[Random Forest]
    D --> G[XGBoost]
    E --> H[["StackingClassifier<br/>ensemble"]]
    F --> H
    G --> H
    H --> I[Evaluation<br/>accuracy, confusion matrix, CV]
    H --> J["Run Prediction<br/>bar-chart visualization"]

    style A fill:#1a2980,color:#fff
    style H fill:#0f6b3a,color:#fff
    style J fill:#26d0ce,color:#000
```

---

## 🗂️ Dataset

The Titanic passenger dataset (`Titanic-Dataset.csv`), containing:
- Passenger demographics (`Age`, `Sex`, `Name`, etc.)
- Ticket information (`Class`, `Fare`, `Cabin`, `Embarked`)
- Survival outcome (`Survived` column: `0` = Not Survived, `1` = Survived)

---

## ✨ Features Used

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Male / Female |
| `Age` | With missing values handled |
| `SibSp` | Siblings/spouses aboard |
| `Parch` | Parents/children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation |
| `Title` | Extracted from `Name` |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | Binary — traveled alone |
| `CabinKnown` | Binary — cabin info available |
| `FarePerPerson` | Fare normalized per family member |

---

## 🔬 Model Training

1. **Data Cleaning** — handle missing values, drop irrelevant columns.
2. **Feature Engineering** — build the derived features above.
3. **Model Selection** — train Logistic Regression, Random Forest, and XGBoost.
4. **Stacking Ensemble** — combine all three for maximum accuracy.
5. **Evaluation** — accuracy, classification report, confusion matrix, cross-validation.

**Expected accuracy: ~88–90%** (best achievable without overfitting).

---

## 📊 Visualization

Every time you hit **Run Prediction**, the model outputs:
- A textual prediction (`Survived` / `Not Survived`)
- A bar chart visualizing survival vs. non-survival probability

---

## ⚡ Usage

1. Clone the repository and place `Titanic-Dataset.csv` in the project folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```
3. Run the script:
   ```bash
   python titanic_prediction.py
   ```
4. Enter passenger details (via form or script input).
5. Click **Run Prediction** → see the survival probability visualization.

---

## 🧪 Example Inputs

<table>
<tr>
<th>✅ Likely to Survive</th>
<th>❌ Likely Not to Survive</th>
</tr>
<tr>
<td>

- Sex: Female
- Age: 22
- Class: 1st
- Fare: 80.00
- FamilySize: 1
- Embarked: Cherbourg
- CabinKnown: Yes

</td>
<td>

- Sex: Male
- Age: 35
- Class: 3rd
- Fare: 7.25
- FamilySize: 1
- Embarked: Southampton
- CabinKnown: No

</td>
</tr>
</table>

---

## 🎓 Internship Deliverable

This project demonstrates:
- End-to-end ML pipeline (data → features → model → evaluation → visualization)
- Professional coding practices with clear documentation
- Practical application of ensemble learning for optimized accuracy

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB6423?style=flat-square)
![matplotlib](https://img.shields.io/badge/-matplotlib-11557C?style=flat-square)

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:26d0ce,100:1a2980&height=100&section=footer" width="100%"/>
</div>
