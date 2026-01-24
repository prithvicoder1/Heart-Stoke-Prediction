# ❤️ Heart Disease Prediction System

A professional AI-powered application designed to assess the risk of heart disease based on patient vitals and medical history. This tool uses machine learning to provide preliminary risk evaluations to assist medical professionals and individuals.

## 🚀 Features
- **User-Friendly Interface:** Clean, professional medical-themed UI built with Streamlit.
- **Real-time Analysis:** Instant risk prediction based on input parameters.
- **Comprehensive Inputs:** Considers age, chest pain type, blood pressure, cholesterol, ECG results, and more.
- **Visual Feedback:** Clear, color-coded results (Safe/Elevated Risk) for easy interpretation.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Heart Disease"
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run frontend/ui.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📂 Project Structure

```
Heart Disease/
├── frontend/
│   └── ui.py               # Main application interface
├── heart_columns.pkl       # Feature columns for the model
├── heart_scaler.pkl        # Data scaler
├── knn_heart_model.pkl     # Trained KNN model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── ...
```

## 📋 Input Parameters

| Parameter | Description |
|-----------|-------------|
| **Age** | Patient's age in years |
| **Sex** | Biological sex (M/F) |
| **Chest Pain Type** | ATA, NAP, TA, or ASY |
| **Resting BP** | Resting blood pressure (mm Hg) |
| **Cholesterol** | Serum cholesterol (mg/dL) |
| **Fasting BS** | Fasting blood sugar > 120 mg/dL (Yes/No) |
| **Resting ECG** | Resting electrocardiogram results |
| **Max HR** | Maximum heart rate achieved |
| **Exercise Angina** | Exercise-induced angina (Yes/No) |
| **Oldpeak** | ST depression induced by exercise |
| **ST Slope** | Slope of the peak exercise ST segment |



![image alt](https://github.com/prithvicoder1/Heart-Stoke-Prediction/blob/78b454e272cbcb1bf4d728116d61cf40127ea718/Screenshot.png)

## ⚠️ Disclaimer
This tool is for **educational and preliminary assessment purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
