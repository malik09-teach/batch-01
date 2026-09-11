"""
dashboard.py
-------------
Clinician-facing Streamlit dashboard for the Heart Disease Decision Support System.

Run with:
    streamlit run dashboard.py

Features:
  - Patient data input form (sidebar)
  - Ensemble prediction with confidence score
  - Individual model agreement (DT / RF / NB votes)
  - Plain-language reasoning (SHAP-based top contributing factors)
  - Decision-tree style rule path for transparency
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

st.set_page_config(page_title="Heart Disease Decision Support", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load and validate every artifact used by the prediction and explanation views."""
    artifact_files = {
        "decision tree": "decision_tree.pkl",
        "random forest": "random_forest.pkl",
        "naive bayes": "gaussian_nb.pkl",
        "ensemble": "voting_ensemble.pkl",
        "scaler": "scaler.pkl",
        "SHAP explainer": "shap_explainer.pkl",
        "feature names": "feature_names.json",
    }
    missing = [
        name for name, filename in artifact_files.items()
        if not os.path.isfile(os.path.join(MODEL_DIR, filename))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)}. Expected them in: {MODEL_DIR}"
        )

    def load_model(filename):
        return joblib.load(os.path.join(MODEL_DIR, filename))

    dt = load_model("decision_tree.pkl")
    rf = load_model("random_forest.pkl")
    nb = load_model("gaussian_nb.pkl")
    ensemble = load_model("voting_ensemble.pkl")
    scaler = load_model("scaler.pkl")
    explainer = load_model("shap_explainer.pkl")
    with open(os.path.join(MODEL_DIR, "feature_names.json"), encoding="utf-8") as f:
        feature_names = json.load(f)

    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names.json must contain a non-empty JSON list.")

    return dt, rf, nb, ensemble, scaler, explainer, feature_names

# Updated to match the exact spelling from the training data
FEATURE_NAMES_READABLE = {
    "id": "Patient ID", "dataset": "Dataset Source",
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalch": "Max Heart Rate", "exang": "Exercise Angina", "oldpeak": "ST Depression",
    "slope": "ST Slope", "ca": "Major Vessels (fluoroscopy)", "thal": "Thalassemia"
}

CP_OPTIONS = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
RESTECG_OPTIONS = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
SLOPE_OPTIONS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_OPTIONS = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Reversible Defect (severe)"}


def get_patient_input():
    st.sidebar.header("Patient Clinical Data")

    age = st.sidebar.slider("Age", 20, 90, 55)
    sex = st.sidebar.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", options=list(CP_OPTIONS.keys()),
                               format_func=lambda x: CP_OPTIONS[x])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", options=[1, 0],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.sidebar.selectbox("Resting ECG", options=list(RESTECG_OPTIONS.keys()),
                                    format_func=lambda x: RESTECG_OPTIONS[x])
    thalch = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.radio("Exercise-Induced Angina", options=[1, 0],
                              format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=list(SLOPE_OPTIONS.keys()),
                                  format_func=lambda x: SLOPE_OPTIONS[x])
    ca = st.sidebar.slider("Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia", options=list(THAL_OPTIONS.keys()),
                                 format_func=lambda x: THAL_OPTIONS[x])

    # Added missing keys ('id', 'dataset') and fixed spelling of 'thalch'
    patient = {
        "id": 0, "dataset": 0, "age": age, "sex": sex, "cp": cp, 
        "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg, 
        "thalch": thalch, "exang": exang, "oldpeak": oldpeak, "slope": slope, 
        "ca": ca, "thal": thal
    }
    return patient


def explain_patient(explainer, patient_scaled_df, feature_names, top_n=5):
    shap_values = explainer.shap_values(patient_scaled_df)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])[0]
    else:
        sv = np.asarray(shap_values)
        sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    sv = np.ravel(sv)
    contributions = list(zip(feature_names, sv))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def main():
    st.title(" Heart Disease Decision Support System")
    st.caption(
        "Ensemble ML tool (Decision Tree + Random Forest + Gaussian Naive Bayes) trained on the "
        "UCI Cleveland Heart Disease dataset. **For clinical decision support only — not a substitute "
        "for professional medical judgment.**"
    )

    dt, rf, nb, ensemble, scaler, explainer, feature_names = load_artifacts()
    patient = get_patient_input()

    patient_df = pd.DataFrame([patient])[feature_names]
    patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=feature_names)

    if st.sidebar.button("Run Prediction", type="primary", use_container_width=True):
        proba_ensemble = ensemble.predict_proba(patient_scaled)[0][1]
        pred_ensemble = int(proba_ensemble >= 0.5)

        proba_dt = dt.predict_proba(patient_scaled)[0][1]
        proba_rf = rf.predict_proba(patient_scaled)[0][1]
        proba_nb = nb.predict_proba(patient_scaled)[0][1]

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.subheader("Prediction Result")
            if pred_ensemble == 1:
                st.error(f" **Disease Likely** — Confidence: {proba_ensemble:.1%}")
            else:
                st.success(f"**No Disease Likely** — Confidence: {(1 - proba_ensemble):.1%}")

            st.metric("Ensemble Risk Probability", f"{proba_ensemble:.1%}")

            st.markdown("##### Individual Model Votes")
            vote_df = pd.DataFrame({
                "Model": ["Decision Tree", "Random Forest", "Gaussian Naive Bayes", "Ensemble (Soft Vote)"],
                "Disease Probability": [proba_dt, proba_rf, proba_nb, proba_ensemble]
            })
            st.dataframe(vote_df.style.format({"Disease Probability": "{:.1%}"}), hide_index=True,
                         use_container_width=True)

        with col2:
            st.subheader("Clinical Reasoning (Top Contributing Factors)")
            top_factors = explain_patient(explainer, patient_scaled, feature_names)
            for feat, val in top_factors:
                readable = FEATURE_NAMES_READABLE.get(feat, feat)
                # Ignore the dummy features we passed in for visualization 
                if feat in ["id", "dataset"]:
                    continue
                direction = "🔺 increases" if val > 0 else "🔻 decreases"
                st.write(f"**{readable}** {direction} risk  `(SHAP impact: {val:+.3f})`")

            st.markdown("---")
            st.markdown("##### Decision Tree Path (transparency view)")
            st.caption("Simplified rule logic the Decision Tree model followed for a patient with similar values:")
            leaf_id = dt.apply(patient_scaled)[0]
            st.code(f"Decision Tree assigned this patient to leaf node #{leaf_id}. "
                    f"See full rule set in reports/decision_tree_rules.txt for exact thresholds.",
                    language="text")

        st.markdown("---")
        st.caption(
            " This tool provides statistical risk estimates based on historical patient data and should "
            "always be interpreted alongside clinical judgment, patient history, and further diagnostic tests."
        )
    else:
        st.info(" Enter patient clinical data in the sidebar and click **Run Prediction** to see results.")


if __name__ == "__main__":
    main()