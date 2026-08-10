"""
Sales Prediction using Python
==============================
Predicts Sales from advertising spend (TV, Radio, Newspaper).

Pipeline:
1. Load & explore the data
2. Feature selection (correlation + Random Forest importance)
3. Train/compare Linear Regression, Polynomial Regression, Random Forest
4. Evaluate on a held-out test set
5. Save the winning model's coefficients (used by sales_prediction_lab.html)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------
df = pd.read_csv("advertising.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelation with Sales:\n", df.corr()["Sales"].sort_values(ascending=False))

# ---------------------------------------------------------------------
# 2. Feature selection
#    TV (r=0.90) and Radio (r=0.35) both correlate strongly with Sales.
#    Newspaper (r=0.16) barely does, and a Random Forest confirms it
#    contributes ~1% of predictive importance -> drop it.
# ---------------------------------------------------------------------
X_full, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
rf_check = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_full, y)
importances = dict(zip(X_full.columns, rf_check.feature_importances_))
print("\nFeature importances:", importances)

FEATURES = ["TV", "Radio"]
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 3. Train & compare candidate models
# ---------------------------------------------------------------------
results = {}

# Plain linear regression
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
results["Linear (TV+Radio)"] = r2_score(y_test, pred)

# Polynomial regression (captures the TV*Radio synergy effect)
poly = PolynomialFeatures(degree=2, include_bias=False)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)
poly_model = LinearRegression().fit(Xp_train, y_train)
pred_poly = poly_model.predict(Xp_test)
results["Polynomial deg=2 (TV+Radio)"] = r2_score(y_test, pred_poly)

# Random Forest, for comparison
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results["Random Forest (TV+Radio)"] = r2_score(y_test, pred_rf)

print("\nModel comparison (R^2 on test set):")
for name, score in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {name:32s} {score:.4f}")

# ---------------------------------------------------------------------
# 4. Evaluate the winner (Polynomial regression) in detail
# ---------------------------------------------------------------------
rmse = mean_squared_error(y_test, pred_poly) ** 0.5
mae = mean_absolute_error(y_test, pred_poly)
cv_scores = cross_val_score(
    LinearRegression(), poly.fit_transform(X), y, cv=10, scoring="r2"
)

print("\nFinal model: Polynomial Regression (degree 2, TV + Radio)")
print(f"  R^2 (test)        : {r2_score(y_test, pred_poly):.4f}")
print(f"  RMSE (test)       : {rmse:.4f}")
print(f"  MAE (test)        : {mae:.4f}")
print(f"  R^2 (10-fold CV)  : {cv_scores.mean():.4f}")

# ---------------------------------------------------------------------
# 5. Coefficients (these power the interactive HTML dashboard)
# ---------------------------------------------------------------------
names = poly.get_feature_names_out(FEATURES)
print("\nCoefficients:")
print(f"  intercept : {poly_model.intercept_}")
for name, coef in zip(names, poly_model.coef_):
    print(f"  {name:10s}: {coef}")


def predict_sales(tv_spend: float, radio_spend: float) -> float:
    """Predict Sales (in thousands of units) from TV and Radio spend ($ thousands)."""
    x = poly.transform([[tv_spend, radio_spend]])
    return float(poly_model.predict(x)[0])


if __name__ == "__main__":
    example = predict_sales(150, 25)
    print(f"\nExample: TV=$150k, Radio=$25k -> predicted Sales = {example:.2f} (000s)")
