"""
Iris Species Classifier — Streamlit App
CodeAlpha Internship Project

Run with:  streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Iris Specimen Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom styling — herbarium / field-guide theme
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;1,500&family=IBM+Plex+Mono:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(160deg, #131F19 0%, #0D1712 100%);
    }
    section[data-testid="stSidebar"] {
        background: #EBE4D2;
    }
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: #F4EFE1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #20291F !important;
    }
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7C9473;
    }
    .specimen-card {
        background: #EBE4D2;
        border-radius: 8px;
        padding: 24px 28px;
        color: #20291F;
    }
    .species-name {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        font-size: 30px;
        margin-bottom: 2px;
    }
    .mono-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #A9AE9E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model + metadata
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "model" / "iris_model.pkl")
    scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")
    le = joblib.load(BASE_DIR / "model" / "label_encoder.pkl")
    with open(BASE_DIR / "model" / "metadata.json") as f:
        meta = json.load(f)
    df = pd.read_csv(BASE_DIR / "data" / "iris.csv")
    df["species"] = df["species"].str.replace("Iris-", "", regex=False)
    return model, scaler, le, meta, df


model, scaler, le, meta, df = load_artifacts()

SPECIES_COLORS = {
    "setosa": "#7C9473",
    "versicolor": "#B08D57",
    "virginica": "#6C4F94",
}
SPECIES_COMMON = {
    "setosa": "Bristle-pointed iris",
    "versicolor": "Blue flag iris",
    "virginica": "Virginia iris",
}


def predict_iris_measurement(sepal_length, sepal_width, petal_length, petal_width):
    X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X_input)
    pred_idx = model.predict(X_scaled)[0]
    pred_species = le.classes_[pred_idx]
    probs = model.predict_proba(X_scaled)[0]
    return {
        "pred_species": pred_species,
        "pred_idx": pred_idx,
        "probabilities": probs,
        "probability_map": {cls: float(probs[i]) for i, cls in enumerate(le.classes_)},
        "x_scaled": X_scaled,
    }

# --------------------------------------------------------------------------
# Sidebar — measurement inputs
# --------------------------------------------------------------------------
st.sidebar.markdown('<p class="eyebrow">Field measurements (cm)</p>', unsafe_allow_html=True)
st.sidebar.markdown("### Enter specimen data")

stats = meta["feature_stats"]

sepal_length = st.sidebar.slider(
    "Sepal length", stats["sepal_length"]["min"], stats["sepal_length"]["max"],
    stats["sepal_length"]["mean"], 0.1,
)
sepal_width = st.sidebar.slider(
    "Sepal width", stats["sepal_width"]["min"], stats["sepal_width"]["max"],
    stats["sepal_width"]["mean"], 0.1,
)
petal_length = st.sidebar.slider(
    "Petal length", stats["petal_length"]["min"], stats["petal_length"]["max"],
    stats["petal_length"]["mean"], 0.1,
)
petal_width = st.sidebar.slider(
    "Petal width", stats["petal_width"]["min"], stats["petal_width"]["max"],
    stats["petal_width"]["mean"], 0.1,
)

st.sidebar.markdown("---")
preset = st.sidebar.selectbox(
    "Or load a typical specimen",
    ["— none —", "Typical setosa", "Typical versicolor", "Typical virginica"],
)
PRESETS = {
    "Typical setosa": (5.0, 3.4, 1.5, 0.2),
    "Typical versicolor": (5.9, 2.8, 4.3, 1.3),
    "Typical virginica": (6.6, 3.0, 5.6, 2.1),
}
if preset != "— none —":
    st.sidebar.info(f"Preset values: {PRESETS[preset]} — set the sliders above to match.")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="eyebrow">CodeAlpha Internship · Species Classification</p>', unsafe_allow_html=True)
st.title("🌸 Iris Specimen Classifier")
st.markdown(
    f'<p class="mono-badge">Model: <b>{meta["model_name"]}</b> &nbsp;·&nbsp; '
    f'10-fold CV accuracy: <b>{meta["full_dataset_cv_accuracy"]:.1%}</b> &nbsp;·&nbsp; '
    f'Trained on <b>150</b> specimens</p>',
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
prediction_payload = predict_iris_measurement(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
)

prediction = prediction_payload
pred_species = prediction["pred_species"]
probs = prediction["probabilities"]

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="specimen-card">', unsafe_allow_html=True)
    st.markdown(f'<p class="eyebrow">Genus Iris · predicted species</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="species-name" style="color:{SPECIES_COLORS[pred_species]}">'
        f'Iris {pred_species}</p>',
        unsafe_allow_html=True,
    )
    st.caption(SPECIES_COMMON[pred_species])

    for i, cls in enumerate(le.classes_):
        st.write(f"**{cls}**")
        st.progress(float(probs[i]), text=f"{probs[i]:.1%}")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    fig = go.Figure()
    categories = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    input_vals = [sepal_length, sepal_width, petal_length, petal_width]

    for species in le.classes_:
        subset = df[df["species"] == species]
        means = [subset[c].mean() for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=means + [means[0]],
            theta=[c.replace("_", " ") for c in categories] + [categories[0].replace("_", " ")],
            fill="toself",
            name=f"{species} (avg)",
            line=dict(color=SPECIES_COLORS[species]),
            opacity=0.35,
        ))

    fig.add_trace(go.Scatterpolar(
        r=input_vals + [input_vals[0]],
        theta=[c.replace("_", " ") for c in categories] + [categories[0].replace("_", " ")],
        fill="toself",
        name="your specimen",
        line=dict(color="#F4EFE1", width=3),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, color="#C7CABE"),
            angularaxis=dict(color="#F4EFE1"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F4EFE1"),
        legend=dict(orientation="h", y=-0.15),
        height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("")
st.markdown("---")

# --------------------------------------------------------------------------
# Model performance section
# --------------------------------------------------------------------------
st.subheader("Model performance")

perf_col1, perf_col2, perf_col3 = st.columns(3)
perf_col1.metric("10-fold CV accuracy (full dataset)", f"{meta['full_dataset_cv_accuracy']:.1%}")
perf_col2.metric("Held-out test accuracy", f"{meta['test_accuracy']:.1%}")
perf_col3.metric("Cross-validated training accuracy", f"{meta['cv_accuracy']:.1%}")

st.write("**Algorithms compared (10-fold stratified cross-validation):**")
comp_df = pd.DataFrame(
    [{"Model": k, "CV Accuracy": v} for k, v in meta["model_comparison"].items()]
).sort_values("CV Accuracy", ascending=False)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.write("**Feature importance:**")
imp_df = pd.DataFrame(
    [{"Feature": k, "Importance": v} for k, v in meta["feature_importance"].items()]
).sort_values("Importance", ascending=False)
st.bar_chart(imp_df.set_index("Feature"))

with st.expander("See EDA plots"):
    img_cols = st.columns(2)
    reports_dir = BASE_DIR / "reports"
    images = [
        "pairplot.png", "correlation_heatmap.png", "feature_boxplots.png",
        "feature_importance.png", "model_comparison.png", "confusion_matrix_full_cv.png",
    ]
    for i, img_name in enumerate(images):
        img_path = reports_dir / img_name
        if img_path.exists():
            img_cols[i % 2].image(str(img_path), caption=img_name.replace("_", " ").replace(".png", ""))

st.caption(
    "Setosa is linearly separable and classified with 100% accuracy. "
    "Nearly all model error occurs between versicolor and virginica, whose petal "
    "measurements genuinely overlap — a property of the data, not a model weakness."
)
