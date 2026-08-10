"""
Titanic Survival Prediction - Optimized Project
Author: Jishnu Vardhan Kancharla
Internship Task 1

Step 8: Visualization on Prediction (standalone console demo).
Run:  python predict_with_visualization.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib.pyplot as plt
import pandas as pd

from model_pipeline import ENCODERS, METRICS, build_model


def run_prediction_with_visualization(model, passenger_data, feature_names):
    """
    Takes passenger input, runs prediction, and shows visualization.
    """
    # Convert passenger data into DataFrame
    input_df = pd.DataFrame([passenger_data], columns=feature_names)

    # Predict probability
    prob = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]

    # Print textual output immediately
    print("\n----------------------------------------")
    if prediction == 1:
        print("✅ Prediction: Passenger is likely to SURVIVE")
    else:
        print("❌ Prediction: Passenger is likely NOT to survive")
    print(f"📊 Probability Survived: {prob[1]:.2%}")
    print(f"📊 Probability Perished: {prob[0]:.2%}")
    print("----------------------------------------\n")

    # Plot survival vs non-survival
    labels = ["Not Survived (0)", "Survived (1)"]
    colors = ["red", "green"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, prob, color=colors)
    plt.title("Titanic Survival Prediction")
    plt.ylabel("Probability")
    plt.ylim(0, 1)

    # Annotate values
    for i, v in enumerate(prob):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plt.savefig("prediction_result.png")
        print("Chart saved to prediction_result.png")



if __name__ == "__main__":
    print("Training the stacking ensemble...")
    stack = build_model()
    print(f"Accuracy: {METRICS['accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {METRICS['cv_score']:.4f}\n")

    # Example passenger input (replace with form values).
    # Categorical values are encoded with the same LabelEncoders used in training:
    # Sex: 1 = male, 0 = female | Embarked: 0=Cherbourg, 1=Queenstown, 2=Southampton
    example_passenger = {
        "Pclass": 3,
        "Sex": int(ENCODERS["Sex"].transform(["male"])[0]),
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": int(ENCODERS["Embarked"].transform(["S"])[0]),
        "Title": int(ENCODERS["Title"].transform(["Mr"])[0]),
        "FamilySize": 1,
        "IsAlone": 1,
        "CabinKnown": 0,
        "FarePerPerson": 7.25
    }

    # Run prediction with visualization
    run_prediction_with_visualization(stack, list(example_passenger.values()), list(example_passenger.keys()))
