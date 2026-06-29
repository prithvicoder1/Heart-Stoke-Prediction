import streamlit as st
import pandas as pd
import joblib
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── STYLES ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f5f4;
    color: #1c1917;
}

/* Hide Streamlit chrome */
header, footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Page wrapper */
.block-container {
    padding: 3rem 1rem 4rem;
    max-width: 780px;
}

/* ── Page header ── */
.page-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.page-header h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1c1917;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem;
}
.page-header p {
    font-size: 0.9rem;
    color: #78716c;
    margin: 0;
}

/* ── Card ── */
[data-testid="stForm"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e7e5e4;
    padding: 2rem 2rem 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a8a29e;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #f5f5f4;
}

/* ── Widget tweaks ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #44403c !important;
    margin-bottom: 2px !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 8px !important;
    border-color: #d6d3d1 !important;
    font-size: 0.875rem !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] {
    padding: 0 !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    width: 100%;
    height: 48px;
    border: none;
    border-radius: 8px;
    background: #1d4ed8;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    cursor: pointer;
    margin-top: 0.5rem;
    transition: background 0.15s ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #1e40af;
}

[data-testid="stFormSubmitButton"] button:active {
    background: #1e3a8a;
    transform: translateY(1px);
}

/* ── Divider inside form ── */
.form-divider {
    height: 1px;
    background: #f5f5f4;
    margin: 1.25rem 0 1.5rem;
}

/* ── Result cards ── */
.result-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e7e5e4;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

.result-card.danger {
    border-top: 3px solid #dc2626;
}

.result-card.safe {
    border-top: 3px solid #059669;
}

.result-card .result-title {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.4rem;
}

.result-card.danger .result-title { color: #b91c1c; }
.result-card.safe  .result-title  { color: #047857; }

.result-card .result-body {
    font-size: 0.85rem;
    color: #57534e;
    margin: 0;
    line-height: 1.6;
}

/* ── Disclaimer ── */
.disclaimer {
    font-size: 0.75rem;
    color: #a8a29e;
    text-align: center;
    margin-top: 1.25rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model   = joblib.load(os.path.join(base_dir, "knn_heart_model.pkl"))
        scaler  = joblib.load(os.path.join(base_dir, "heart_scaler.pkl"))
        cols    = joblib.load(os.path.join(base_dir, "heart_columns.pkl"))
        return model, scaler, cols
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


model, scaler, expected_columns = load_assets()

if model is None:
    st.stop()


# ─── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>Enter patient details below to generate an AI-powered cardiac risk assessment.</p>
</div>
""", unsafe_allow_html=True)


# ─── FORM ────────────────────────────────────────────────────────────────────
with st.form("main_form"):

    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina · NAP: Non-Anginal · TA: Typical Angina · ASY: Asymptomatic"
        )
        resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

    with col2:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            help="ST: ST-T wave abnormality · LVH: Left Ventricular Hypertrophy"
        )
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        ex_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Generate Assessment")


# ─── PREDICTION ──────────────────────────────────────────────────────────────
if submitted:

    input_data = {
        "Age":           age,
        "RestingBP":     resting_bp,
        "Cholesterol":   cholesterol,
        "FastingBs":     fasting_bs,
        "MaxHR":         max_hr,
        "Oldpeak":       oldpeak,
        f"Sex_{sex}":                    1,
        f"ChestPainType_{chest_pain}":   1,
        f"RestingECG_{resting_ecg}":     1,
        f"ExerciseAngina_{ex_angina}":   1,
        f"ST_Slope_{st_slope}":          1,
    }

    df = pd.DataFrame([input_data])

    if expected_columns is None or scaler is None:
        st.error("Model files were not loaded correctly.")
        st.stop()

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=expected_columns, fill_value=0)

    try:
        scaled     = scaler.transform(df)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-card danger">
                <p class="result-title">⚠️ Elevated Risk Detected</p>
                <p class="result-body">
                    The model indicates an increased likelihood of heart disease based on the
                    provided parameters. Please refer this patient to a qualified cardiologist
                    for further clinical evaluation and diagnostics.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card safe">
                <p class="result-title">✅ Low Risk</p>
                <p class="result-body">
                    The model predicts a low likelihood of heart disease at this time.
                    Encourage the patient to maintain a healthy lifestyle, balanced diet,
                    and schedule regular preventive health check-ups.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

    st.markdown("""
    <p class="disclaimer">
        This tool is for informational purposes only and does not constitute medical advice.<br>
        Always consult a licensed healthcare professional for diagnosis and treatment.
    </p>
    """, unsafe_allow_html=True)