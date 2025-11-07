# Minimal Streamlit app to view README, train sample models, and run predictions.
import streamlit as st
from pathlib import Path
import pandas as pd
from src import modeling

ROOT = Path(__file__).parent
README = ROOT / "README.md"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config("AI Workflow â€” Models", layout="wide")
st.title("AI Development Workflow â€” Example Models")

with st.expander("Assignment README (click to view)"):
    st.markdown(README.read_text(encoding="utf-8"))

st.sidebar.header("Actions")
action = st.sidebar.selectbox("Action", ["Use sample student data", "Use sample readmission data", "Upload CSV (predict)"])

if action == "Upload CSV (predict)":
    uploaded = st.sidebar.file_uploader("Upload CSV with features", type=["csv"])
    model_choice = st.sidebar.selectbox("Model to apply", [p.name for p in MODELS_DIR.glob("*.joblib")] or ["(train a model first)"])
    if uploaded and st.sidebar.button("Run prediction"):
        df = pd.read_csv(uploaded)
        if model_choice == "(train a model first)":
            st.sidebar.error("No model available. Train a model using the sample actions first.")
        else:
            model = modeling.load_model(MODELS_DIR / model_choice)
            numeric = df.select_dtypes(include=["int64", "float64"])
            probs = modeling.predict(model, numeric)
            out = df.copy()
            out["risk_score"] = probs
            st.dataframe(out.head(200))
            st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

elif action == "Use sample student data":
    st.header("Student Dropout â€” Sample")
    X, y = modeling.synth_student_data(n=1000)
    st.write("Sample features:")
    st.dataframe(X.head())
    if st.button("Train student RandomForest"):
        res = modeling.train_student_model(X, y)
        st.success(f"Trained. Model saved to: {res['model_path']}")
        st.json(res["report"])

elif action == "Use sample readmission data":
    st.header("Hospital Readmission â€” Sample")
    X, y = modeling.synth_readmission_data(n=1000)
    st.write("Sample features:")
    st.dataframe(X.head())
    if st.button("Train readmission GradientBoosting"):
        res = modeling.train_readmission_model(X, y)
        st.success(f"Trained. Model saved to: {res['model_path']}")
        st.json(res["report"])

st.sidebar.markdown("Model files saved in ./models/*.joblib")