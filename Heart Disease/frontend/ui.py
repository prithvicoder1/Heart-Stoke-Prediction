import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Heart Stroke Prediction", layout="wide", page_icon="❤️")

# Custom CSS for futuristic styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;700&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d2ff;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.7);
    }
    
    h1 {
        text-align: center;
        padding-bottom: 20px;
        font-size: 3.5rem;
    }
    
    /* Glassmorphism Containers */
    .stMarkdown, .stColumn {
        /* We can't easily target the column container directly with CSS injection in Streamlit, 
           but we can style the content within. */
    }
    
    /* Custom container styling hack */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Input Fields Styling */
    .stSelectbox > div > div > div, .stNumberInput > div > div > input {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #00d2ff !important;
        border: 1px solid #00d2ff !important;
        border-radius: 8px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #00d2ff !important;
    }
    
    /* Labels */
    .stMarkdown p, label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #e0e0e0 !important;
        font-size: 1.1rem;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border-radius: 30px;
        padding: 15px 40px;
        border: none;
        font-size: 18px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(0, 210, 255, 0.8);
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
    }
    
    /* Success/Error Messages */
    .stAlert {
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border-radius: 10px;
    }
    
    </style>
""", unsafe_allow_html=True)

# Load models from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

model_path = os.path.join(parent_dir, "knn_heart_model.pkl")
scaler_path = os.path.join(parent_dir, "heart_scaler.pkl")
columns_path = os.path.join(parent_dir, "heart_columns.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    expected_columns = joblib.load(columns_path)
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("Heart Stroke Prediction ❤️")
st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 1.2rem;'>Advanced AI Health Assessment System</p>", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Details")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    Chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.markdown("### 🩺 Medical Indicators")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

if st.button("Predict Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBs': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + Chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    
    input_df = pd.DataFrame([raw_input])
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.markdown("**Action Required:** Please consult a cardiologist for a detailed examination.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.markdown("**Status:** Your heart health indicators appear normal. Keep up the good work!")
