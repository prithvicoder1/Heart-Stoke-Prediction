import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f0f8ff 0%, #ffffff 100%);
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
    }
    div.stButton > button {
        background: linear-gradient(to right, #4facfe, #00f2fe);
        color: white;
        border: none;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    [data-testid="stForm"] {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    .safe { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_assets():
    try:
        # Since this app.py is in the root, we look in the current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(base_dir, "knn_heart_model.pkl"))
        scaler = joblib.load(os.path.join(base_dir, "heart_scaler.pkl"))
        cols = joblib.load(os.path.join(base_dir, "heart_columns.pkl"))
        return model, scaler, cols
    except Exception as e:
        return None, None, None

model, scaler, expected_columns = load_assets()

if not model:
    st.error("❌ System Error: Model files missing. Please ensure .pkl files are in the same directory.")
    st.stop()

# --- UI Layout ---
st.title("❤️ Heart Disease Prediction")
st.markdown("<p style='text-align: center; color: #57606f;'>AI-Powered Professional Health Assessment</p>", unsafe_allow_html=True)

with st.form("main_form"):
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        age = st.number_input("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        ex_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown("###")
    submitted = st.form_submit_button("Generate Assessment")

# --- Logic ---
if submitted:
    input_data = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBs': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
        f'Sex_{sex}': 1, f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1, f'ExerciseAngina_{ex_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }
    
    # Process Data
    df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in df.columns: df[col] = 0
    df = df[expected_columns]

    # Predict
    try:
        pred = model.predict(scaler.transform(df))[0]
        
        if pred == 1:
            st.markdown("""
                <div class="result-box danger">
                    <h3>⚠️ Elevated Risk Detected</h3>
                    <p>Consult a cardiologist for a comprehensive check-up.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box safe">
                    <h3>✅ Low Risk Profile</h3>
                    <p>Maintain a healthy lifestyle and regular check-ups.</p>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
