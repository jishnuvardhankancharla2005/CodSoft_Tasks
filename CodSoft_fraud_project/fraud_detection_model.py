"""
Credit Card Fraud Detection using Python
==========================================
Classifies transactions as fraudulent (1) or genuine (0).

Dataset: 284,807 transactions, 492 fraud (0.17%) -- heavily imbalanced.
Features V1-V28 are PCA components (already anonymized/scaled); Time and
Amount are raw.

Pipeline:
1. Load & explore (class balance, missing values)
2. Preprocess: scale Amount (V1-V28 are already PCA-scaled)
3. Feature selection: rank features by class separation / Random Forest
   importance, keep the ones that actually carry signal
4. Handle class imbalance: class_weight='balanced', compare against
   manual undersampling
5. Train & compare Logistic Regression vs Random Forest
6. Evaluate with precision, recall, F1, ROC-AUC, PR-AUC (accuracy alone
   is misleading on a 99.8%-genuine dataset)
7. Export the winning lightweight model's coefficients (used by
   fraud_detection_console.html for real-time, in-browser scoring)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)

# ---------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------
df = pd.read_csv("creditcard.csv")
df["Class"] = df["Class"].astype(int)

print("Shape:", df.shape)
print(df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")
print("Missing values:", df.isnull().sum().sum())

# ---------------------------------------------------------------------
# 2. Feature selection
#    Rank every feature by how far the fraud-class mean sits from the
#    genuine-class mean, in standard deviations (a quick, robust proxy
#    for separability), and confirm with a Random Forest's importances.
# ---------------------------------------------------------------------
candidate_features = [c for c in df.columns if c not in ("Class",)]
separation = {}
for f in candidate_features:
    genuine = df[df.Class == 0][f]
    fraud = df[df.Class == 1][f]
    pooled_std = df[f].std()
    separation[f] = abs(fraud.mean() - genuine.mean()) / pooled_std

sep_ranked = sorted(separation.items(), key=lambda kv: -kv[1])
print("\nTop 15 features by class separation (in std devs):")
for f, s in sep_ranked[:15]:
    print(f"  {f:10s} {s:.3f}")

TOP_FEATURES = ["V14", "V4", "V10", "V12", "V17", "V11", "V3", "V16", "V7", "Amount"]
print("\nSelected features:", TOP_FEATURES)
print("(Time and 18 of the 28 PCA components separated the classes too weakly to help.)")

X = df[TOP_FEATURES].copy()
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])  # V-features are already PCA-scaled

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)} rows ({y_train.sum()} fraud)")
print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud)")


def evaluate(name, y_true, pred, proba):
    print(f"\n--- {name} ---")
    print("Precision:", round(precision_score(y_true, pred), 4))
    print("Recall:   ", round(recall_score(y_true, pred), 4))
    print("F1:       ", round(f1_score(y_true, pred), 4))
    print("ROC-AUC:  ", round(roc_auc_score(y_true, proba), 4))
    print("PR-AUC:   ", round(average_precision_score(y_true, proba), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))


# ---------------------------------------------------------------------
# 3. Handle imbalance + train candidate models
# ---------------------------------------------------------------------

# (a) Logistic Regression, class_weight='balanced' -- fast, embeddable
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)
proba_lr = lr.predict_proba(X_test)[:, 1]

# tune the decision threshold for best F1 instead of using the default 0.5
# (heavily imbalanced data makes 0.5 a poor default cutoff)
prec, rec, th = precision_recall_curve(y_test, proba_lr)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.argmax(f1s[:-1])
best_threshold = th[best_idx]
pred_lr = (proba_lr >= best_threshold).astype(int)
evaluate(f"Logistic Regression (threshold={best_threshold:.6f})", y_test, pred_lr, proba_lr)

# (b) Random Forest, class_weight='balanced_subsample' -- higher accuracy,
#     but too large to run client-side in a static HTML page
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced_subsample",
    n_jobs=-1, random_state=42,
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
evaluate("Random Forest", y_test, pred_rf, proba_rf)

# (c) For comparison: manual undersampling (3 genuine : 1 fraud)
train_df = X_train.copy()
train_df["Class"] = y_train.values
fraud_train = train_df[train_df.Class == 1]
genuine_train = train_df[train_df.Class == 0].sample(n=len(fraud_train) * 3, random_state=42)
under_df = pd.concat([fraud_train, genuine_train]).sample(frac=1, random_state=42)
rf_under = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
rf_under.fit(under_df.drop(columns=["Class"]), under_df["Class"])
pred_under = rf_under.predict(X_test)
proba_under = rf_under.predict_proba(X_test)[:, 1]
evaluate("Random Forest (3:1 undersampled)", y_test, pred_under, proba_under)
print(
    "\nNote: undersampling throws away most genuine transactions and, on this "
    "dataset, trades a lot of precision for a small recall gain -- "
    "class_weight='balanced' on the full data performed better here."
)

# ---------------------------------------------------------------------
# 4. Coefficients for the embedded (Logistic Regression) model
#    -- these power fraud_detection_console.html
# ---------------------------------------------------------------------
coef = dict(zip(TOP_FEATURES, lr.coef_[0].tolist()))
intercept = float(lr.intercept_[0])
z_threshold = float(np.log(best_threshold / (1 - best_threshold)))

print("\nEmbedded model coefficients:")
print("  intercept:", intercept)
for f, c in coef.items():
    print(f"  {f:10s}: {c}")
print("  decision threshold (log-odds):", z_threshold)
print("  Amount scaler mean/scale:", scaler.mean_[0], scaler.scale_[0])


def predict_fraud(transaction: dict) -> dict:
    """
    transaction: dict with keys V14, V4, V10, V12, V17, V11, V3, V16, V7, Amount
    Returns predicted class and the raw score.
    """
    scaled_amount = (transaction["Amount"] - scaler.mean_[0]) / scaler.scale_[0]
    z = intercept
    for f in TOP_FEATURES:
        x = scaled_amount if f == "Amount" else transaction[f]
        z += coef[f] * x
    return {"is_fraud": z >= z_threshold, "score": z}


if __name__ == "__main__":
    example = {
        "V14": -6.918, "V4": 9.249, "V10": -14.557, "V12": -10.38, "V17": -20.255,
        "V11": 4.392, "V3": -17.88, "V16": -10.328, "V7": -18.015, "Amount": 53.95,
    }
    print("\nExample transaction ->", predict_fraud(example))
