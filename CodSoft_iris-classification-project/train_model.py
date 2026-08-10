"""
Iris Species Classification - Model Training Pipeline
CodeAlpha Internship Project

Performs EDA, feature importance analysis, compares multiple classification
algorithms via stratified cross-validation, tunes the best candidate, and
exports the final model + supporting artifacts for the app.
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Load & inspect data
# --------------------------------------------------------------------------
df = pd.read_csv("data/iris.csv")
df["species"] = df["species"].str.replace("Iris-", "", regex=False)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nClass balance:\n{df['species'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
print(f"\nDescribe:\n{df.describe()}")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

# --------------------------------------------------------------------------
# 2. EDA plots
# --------------------------------------------------------------------------
pair = sns.pairplot(df, hue="species", diag_kind="kde", palette="viridis")
pair.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
pair.savefig("reports/pairplot.png", dpi=140, bbox_inches="tight")
plt.close("all")

plt.figure(figsize=(7, 5.5))
corr = df[FEATURES].corr()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=140)
plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(data=df, x="species", y=feat, hue="species", palette="viridis", ax=ax, legend=False)
    ax.set_title(feat.replace("_", " ").title())
plt.tight_layout()
plt.savefig("reports/feature_boxplots.png", dpi=140)
plt.close("all")

# --------------------------------------------------------------------------
# 3. Encode + split
# --------------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[FEATURES].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 4. Feature importance (Random Forest as a fast, reliable importance proxy)
# --------------------------------------------------------------------------
rf_importance = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
rf_importance.fit(X_train, y_train)
importances = pd.Series(rf_importance.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)
print(importances)

plt.figure(figsize=(7, 4.5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
plt.title("Feature Importance for Species Classification")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=140)
plt.close("all")

# --------------------------------------------------------------------------
# 5. Compare multiple algorithms with stratified 10-fold CV
# --------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON (10-fold stratified cross-validation, scaled features)")
print("=" * 60)

results = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"{name:25s} | mean acc: {scores.mean():.4f}  std: {scores.std():.4f}")

best_name = max(results, key=lambda k: results[k]["mean"])
print(f"\nBest candidate by CV mean accuracy: {best_name}")

plt.figure(figsize=(8, 4.5))
names = list(results.keys())
means = [results[n]["mean"] for n in names]
stds = [results[n]["std"] for n in names]
bars = plt.bar(names, means, yerr=stds, capsize=5, color=sns.color_palette("viridis", len(names)))
plt.ylim(0.85, 1.01)
plt.ylabel("Cross-Validated Accuracy")
plt.title("Model Comparison — 10-Fold Stratified CV")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=140)
plt.close("all")

# --------------------------------------------------------------------------
# 6. Hyperparameter tuning on the strongest candidates (SVM + KNN + RF)
#    to squeeze out maximum achievable accuracy
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

tuned_results = {}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=RANDOM_STATE),
    param_grid={"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1, 1], "kernel": ["rbf", "linear"]},
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
svm_grid.fit(X_train_scaled, y_train)
tuned_results["SVM"] = (svm_grid.best_estimator_, svm_grid.best_score_, svm_grid.best_params_)
print(f"SVM best: {svm_grid.best_score_:.4f}  params: {svm_grid.best_params_}")

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": list(range(3, 16)), "weights": ["uniform", "distance"]},
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
knn_grid.fit(X_train_scaled, y_train)
tuned_results["KNN"] = (knn_grid.best_estimator_, knn_grid.best_score_, knn_grid.best_params_)
print(f"KNN best: {knn_grid.best_score_:.4f}  params: {knn_grid.best_params_}")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid={"n_estimators": [100, 300], "max_depth": [None, 3, 5, 8], "min_samples_leaf": [1, 2, 4]},
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
rf_grid.fit(X_train_scaled, y_train)
tuned_results["RandomForest"] = (rf_grid.best_estimator_, rf_grid.best_score_, rf_grid.best_params_)
print(f"RandomForest best: {rf_grid.best_score_:.4f}  params: {rf_grid.best_params_}")

best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k][1])
best_model, best_cv_score, best_params = tuned_results[best_tuned_name]
print(f"\nSelected final model: {best_tuned_name}  (CV accuracy: {best_cv_score:.4f})")

# --------------------------------------------------------------------------
# 7. Final evaluation on held-out test set
# --------------------------------------------------------------------------
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"FINAL TEST SET ACCURACY: {test_acc:.4f}")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(5.5, 5))
disp.plot(ax=ax, cmap="viridis", colorbar=False)
plt.title(f"Confusion Matrix — {best_tuned_name} (Test Accuracy: {test_acc:.2%})")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=140)
plt.close("all")

# --------------------------------------------------------------------------
# 7b. More robust accuracy estimate: 10-fold CV predictions across ALL 150
#     samples (a single 30-row holdout is noisy for a dataset this small)
# --------------------------------------------------------------------------
full_scaler = StandardScaler().fit(X)
X_full_scaled = full_scaler.transform(X)
cv_full_preds = cross_val_predict(best_model, X_full_scaled, y, cv=cv)
full_cv_acc = accuracy_score(y, cv_full_preds)

print("\n" + "=" * 60)
print(f"ROBUST ESTIMATE — 10-FOLD CV ACCURACY ACROSS ALL 150 SAMPLES: {full_cv_acc:.4f}")
print("=" * 60)
print(classification_report(y, cv_full_preds, target_names=le.classes_))

cm_full = confusion_matrix(y, cv_full_preds)
disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(5.5, 5))
disp_full.plot(ax=ax, cmap="viridis", colorbar=False)
plt.title(f"Confusion Matrix — 10-Fold CV, Full Dataset (Accuracy: {full_cv_acc:.2%})")
plt.tight_layout()
plt.savefig("reports/confusion_matrix_full_cv.png", dpi=140)
plt.close("all")

# --------------------------------------------------------------------------
# 8. Fit final model on ALL data (train+test) for deployment, save artifacts
# --------------------------------------------------------------------------
final_scaler = StandardScaler().fit(X)
X_all_scaled = final_scaler.transform(X)
best_model.fit(X_all_scaled, y)

joblib.dump(best_model, "model/iris_model.pkl")
joblib.dump(final_scaler, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")

metadata = {
    "model_name": best_tuned_name,
    "best_params": best_params,
    "cv_accuracy": round(best_cv_score, 4),
    "test_accuracy": round(test_acc, 4),
    "full_dataset_cv_accuracy": round(full_cv_acc, 4),
    "features": FEATURES,
    "classes": list(le.classes_),
    "feature_importance": importances.round(4).to_dict(),
    "model_comparison": {k: round(v["mean"], 4) for k, v in results.items()},
    "feature_stats": {
        feat: {
            "min": round(float(df[feat].min()), 2),
            "max": round(float(df[feat].max()), 2),
            "mean": round(float(df[feat].mean()), 2),
        }
        for feat in FEATURES
    },
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model, scaler, label encoder, and metadata to /model")
print("Saved EDA + evaluation plots to /reports")
