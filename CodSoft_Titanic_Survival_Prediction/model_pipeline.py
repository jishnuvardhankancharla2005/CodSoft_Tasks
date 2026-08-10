"""
Titanic Survival Prediction - Optimized Project
Author: Jishnu Vardhan Kancharla
Internship Task 1

Shared model pipeline (dataset loading, feature engineering, training).
Used by both the Flask web dashboard (app.py) and the console
visualization script (predict_with_visualization.py).
"""

import os
import urllib.request
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_FILE = "Titanic-Dataset.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


# ================================
# Step 2: Load Dataset (auto-download if missing)
# ================================
def ensure_dataset():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found locally, downloading a public mirror...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
            print("Downloaded", DATA_FILE)
        except Exception as exc:
            raise RuntimeError(
                "Could not find or download the Titanic dataset. "
                "Place 'Titanic-Dataset.csv' next to the script.\n" + str(exc)
            ) from exc


def load_dataset():
    ensure_dataset()
    return pd.read_csv(DATA_FILE)


# ================================
# Step 3-4: Feature engineering
# ================================
def feature_engineering(df):
    df = df.copy()

    # Extract Title from Name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin presence
    df["CabinKnown"] = df["Cabin"].notnull().astype(int)

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FarePerPerson"] = df["FarePerPerson"].fillna(0.0)

    return df


# ================================
# Global state (built at startup)
# ================================
MODEL = None
ENCODERS = {}
X_COLUMNS = None
METRICS = {}
TRAIN_DF = None


# ================================
# Step 5-7: Train, evaluate, return model
# ================================
def build_model():
    global MODEL, ENCODERS, X_COLUMNS, METRICS, TRAIN_DF

    raw = load_dataset()
    df = feature_engineering(raw)

    # Encode categorical variables
    for col in ["Sex", "Embarked", "Title"]:
        ENCODERS[col] = LabelEncoder().fit(df[col])
        df[col] = ENCODERS[col].transform(df[col])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_COLUMNS = list(X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build models
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss",
    )

    # Stacking Ensemble
    stack = StackingClassifier(
        estimators=[("lr", log_reg), ("rf", rf)],
        final_estimator=xgb,
        cv=5,
    )

    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    METRICS["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    METRICS["report"] = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    METRICS["confusion"] = cm.tolist()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    METRICS["cv_score"] = round(cross_val_score(stack, X, y, cv=cv).mean(), 4)

    # Feature importance from the fitted RandomForest base estimator,
    # which runs on the original 12 features (the XGBoost meta-learner
    # only sees 2 stacking inputs, so its importances would be misleading).
    importances = stack.estimators_[1].feature_importances_
    METRICS["importance"] = sorted(
        (float(i), col) for col, i in zip(X_COLUMNS, importances)
    )[::-1]

    # Dataset stats for the dashboard charts
    TRAIN_DF = df.assign(Survived=y)
    MODEL = stack
    return stack


def _predict_row(payload):
    """Convert a JSON passenger payload into a model-ready feature row."""
    sex = str(payload.get("sex", "male")).strip().lower()
    if sex not in ("male", "female"):
        sex = "male"

    name = str(payload.get("name", "")).strip()
    title = None
    if name:
        m = pd.Series([name]).str.extract(r" ([A-Za-z]+)\.")[0]
        title = m.values[0] if m.notna().any() else None
    if title:
        title = title.replace("Mlle", "Miss").replace("Ms", "Miss").replace("Mme", "Mrs")
        if title not in ENCODERS["Title"].classes_:
            title = "Rare"
    else:
        title = "Mr" if sex == "male" else "Miss"

    sibsp = int(payload.get("sibsp", 0))
    parch = int(payload.get("parch", 0))
    family_size = sibsp + parch + 1
    fare = float(payload.get("fare", 0.0))
    fare_per_person = fare / family_size if family_size else 0.0

    embarked = str(payload.get("embarked", "S")).upper()
    if "Embarked" in ENCODERS and embarked not in ENCODERS["Embarked"].classes_:
        embarked = "S"

    row = {
        "Pclass": int(payload.get("pclass", 3)),
        "Sex": ENCODERS["Sex"].transform([sex])[0],
        "Age": float(payload.get("age", 30)),
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": ENCODERS["Embarked"].transform([embarked])[0],
        "Title": ENCODERS["Title"].transform([title])[0],
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "CabinKnown": int(bool(payload.get("cabin_known", False))),
        "FarePerPerson": fare_per_person,
    }
    return pd.DataFrame([row], columns=X_COLUMNS)

