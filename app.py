from pathlib import Path

import pandas as pd
import streamlit as st

from drug_system import FEATURE_NAMES, load_artifact, load_metrics, load_or_create_dataset, predict_batch, predict_single, train_and_save

st.set_page_config(page_title="Drug Recommendation System", page_icon="🧪", layout="wide")

st.title("Drug Recommendation System")
st.caption("Educational demo that predicts a likely drug class from patient features.")

base_dir = Path(__file__).resolve().parent
artifact = load_artifact()
metrics = load_metrics()
dataset = load_or_create_dataset()

drug_descriptions = {
    "DrugA": "Recommended more often for younger patients with high blood pressure and high cholesterol.",
    "DrugB": "Recommended more often for older patients with stronger cardiovascular risk patterns.",
    "DrugC": "Recommended more often when blood pressure is low.",
    "DrugX": "Recommended more often for moderate cases without high sodium-to-potassium ratio.",
    "DrugY": "Recommended more often when sodium-to-potassium ratio is high.",
}

with st.sidebar:
    st.header("Model Actions")
    if st.button("Retrain Model", use_container_width=True):
        artifact, metrics = train_and_save()
        st.success("Model retrained and artifacts updated.")

    st.header("Model Summary")
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

left_column, right_column = st.columns([1.1, 1])

with left_column:
    st.subheader("Single Patient Recommendation")
    with st.form("patient_form"):
        age = st.slider("Age", min_value=18, max_value=80, value=45)
        sex = st.selectbox("Sex", options=["F", "M"])
        blood_pressure = st.selectbox("Blood Pressure", options=["LOW", "NORMAL", "HIGH"])
        cholesterol = st.selectbox("Cholesterol", options=["NORMAL", "HIGH"])
        na_to_k = st.slider("Sodium to Potassium Ratio", min_value=6.0, max_value=38.0, value=16.5, step=0.1)
        submitted = st.form_submit_button("Recommend Drug", use_container_width=True)

    if submitted:
        result = predict_single(
            {
                "age": age,
                "sex": sex,
                "blood_pressure": blood_pressure,
                "cholesterol": cholesterol,
                "na_to_k": na_to_k,
            },
            artifact=artifact,
        )
        st.metric("Recommended Drug", result["recommended_drug"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        st.info(drug_descriptions[result["recommended_drug"]])

        probability_frame = pd.DataFrame(
            {
                "drug": list(result["probabilities"].keys()),
                "probability": list(result["probabilities"].values()),
            }
        ).sort_values("probability", ascending=False)
        st.bar_chart(probability_frame.set_index("drug"))

with right_column:
    st.subheader("Batch CSV Scoring")
    st.write("Upload a CSV with the required feature columns to score multiple patients.")
    uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
    if uploaded_file is not None:
        batch_frame = pd.read_csv(uploaded_file)
        missing_columns = [column for column in FEATURE_NAMES if column not in batch_frame.columns]
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        else:
            scored_frame = predict_batch(batch_frame, artifact=artifact)
            st.dataframe(scored_frame.head(20), use_container_width=True)
            st.download_button(
                "Download Scored CSV",
                scored_frame.to_csv(index=False).encode("utf-8"),
                file_name="drug_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.subheader("Sample Dataset Preview")
    st.dataframe(dataset.head(10), use_container_width=True)

st.subheader("Artifacts")
st.write(f"Model artifact: `{base_dir / 'models' / 'drug_model.joblib'}`")
st.write(f"Metrics report: `{base_dir / 'reports' / 'metrics.json'}`")
st.write(f"Confusion matrix image: `{base_dir / 'reports' / 'confusion_matrix.png'}`")
st.warning("This project is for education and demos only. It should not be used as real medical advice.")
