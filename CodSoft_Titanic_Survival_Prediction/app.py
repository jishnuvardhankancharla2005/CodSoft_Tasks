"""
Titanic Survival Prediction - Optimized Project
Author: Jishnu Vardhan Kancharla
Internship Task 1

Flask backend that trains the stacking ensemble (LogisticRegression + RandomForest
stacked on XGBoost) and serves it behind an animated web dashboard,
including the Step 8 survival-probability chart on every prediction.
"""

import base64
import io
import sys
import threading
import webbrowser

import matplotlib
matplotlib.use("Agg")  # headless backend so charts render inside the web server
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, render_template, request

import model_pipeline
from model_pipeline import build_model

app = Flask(__name__)


def _ensure_model():
    if model_pipeline.MODEL is None:
        model_pipeline.build_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    _ensure_model()
    df = model_pipeline.TRAIN_DF
    survived = df["Survived"]


    def breakdown(group_col):
        out = {}
        counts = df.groupby(group_col)["Survived"].value_counts().unstack(fill_value=0)
        for key, row in counts.iterrows():
            label = (
                str(key)
                if not isinstance(key, pd.Interval)
                else f"{int(key.left)}-{int(key.right)}"
            )
            out[label] = {
                "survived": int(row.get(1, 0)),
                "not_survived": int(row.get(0, 0)),
            }
        return out

    sex_survival = {}
    sex_counts = df.groupby("Sex")["Survived"].value_counts().unstack(fill_value=0)
    for key, row in sex_counts.iterrows():
        sex_survival[str(model_pipeline.ENCODERS["Sex"].inverse_transform([int(key)])[0])] = int(
            row.get(1, 0)
        )

    return jsonify(
        {
            "total": int(len(df)),
            "survived": int(survived.sum()),
            "not_survived": int((survived == 0).sum()),
            "survival_rate": round(float(survived.mean()), 4),
            "accuracy": model_pipeline.METRICS["accuracy"],
            "cv_score": model_pipeline.METRICS["cv_score"],
            "importance": [
                {"feature": col, "value": val}
                for val, col in model_pipeline.METRICS["importance"][:8]
            ],
            "pclass_survival": breakdown("Pclass"),
            "sex_survival": sex_survival,
            "age_bins": [
                {"label": k, **v}
                for k, v in breakdown(
                    pd.cut(df["Age"], bins=[0, 10, 20, 30, 40, 60, 100])
                ).items()
            ],
            "confusion": model_pipeline.METRICS["confusion"],
        }
    )


# ================================
# Step 8: Visualization on Prediction
# ================================
def prediction_chart(proba, pred):
    """Render the survival-vs-not bar chart (Step 8) and return a base64 PNG."""
    labels = ["Not Survived (0)", "Survived (1)"]
    colors = ["#f87171", "#34d399"]

    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="#0c2238")
    ax.set_facecolor("#0c2238")
    bars = ax.bar(labels, proba, color=colors, width=0.5)
    ax.set_title("Titanic Survival Prediction", color="#e8f1f8", fontsize=13, fontweight="bold")
    ax.set_ylabel("Probability", color="#9fb7c9")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#9fb7c9")
    for spine in ax.spines.values():
        spine.set_color((0.47, 0.67, 0.82, 0.4))
    for i, v in enumerate(proba):
        ax.text(
            i, v + 0.02, f"{v:.2f}", ha="center",
            fontweight="bold", color="#e8f1f8",
        )
        bars[i].set_edgecolor("#0c2238")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


@app.route("/api/predict", methods=["POST"])
def predict():
    _ensure_model()
    payload = request.get_json(force=True)

    row = model_pipeline._predict_row(payload)
    proba = model_pipeline.MODEL.predict_proba(row)[0]
    pred = int(model_pipeline.MODEL.predict(row)[0])
    return jsonify(
        {
            "prediction": pred,
            "label": "Survived" if pred == 1 else "Did Not Survive",
            "probability": round(float(proba[pred]), 4),
            "probability_survived": round(float(proba[1]), 4),
            "chart": prediction_chart(proba, pred),
        }
    )


if __name__ == "__main__":
    try:
        build_model()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print("\nModel ready.")
    print("  Accuracy:", model_pipeline.METRICS["accuracy"])
    print("  Cross-validation Accuracy:", model_pipeline.METRICS["cv_score"])
    print("  Dashboard: http://127.0.0.1:5000\n")

    # Open the browser automatically once the server is listening.
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)
