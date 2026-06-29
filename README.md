# ❤️ HeartVision-AI

<p align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563eb,100:06b6d4&height=220&section=header&text=HeartVision%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn"/>

</p>

<p align="center">

AI-Powered Heart Disease Risk Prediction using Machine Learning, Streamlit & Docker

</p>

<p align="center">

<a href="https://heartvision-ai.onrender.com/"><img src="https://img.shields.io/badge/Live%20Demo-HeartVision%20AI-2563eb?style=for-the-badge"></a>

<a href="https://github.com/prithvicoder1/HeartVision-AI"><img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github"></a>

</p>

---

## ✨ Overview

HeartVision-AI is a machine learning web application that predicts the likelihood of heart disease from clinical parameters. It provides a clean Streamlit interface, real-time predictions, Docker support, and deployment on Render.

### 🚀 Features

- ❤️ AI-powered heart disease prediction

- 📊 Real-time inference using a trained KNN model

- 🖥️ Modern Streamlit interface

- 🧠 Preprocessing with saved scaler & feature columns

- 🐳 Docker support

- ☁️ Render deployment ready

- 📱 Responsive UI

---

## 🌐 Live Demo

**https://heartvision-ai.onrender.com/**

## 📦 Repository

**https://github.com/prithvicoder1/HeartVision-AI**

---

## 🛠️ Tech Stack

| Category | Technology |

|-----------|------------|

| Language | Python |

| ML | Scikit-learn |

| UI | Streamlit |

| Data | Pandas, NumPy |

| Model Storage | Joblib |

| Deployment | Render |

| Containerization | Docker |

---

## 📂 Project Structure

```text

HeartVision-AI/

├── app.py

├── Dockerfile

├── requirements.txt

├── Procfile

├── runtime.txt

├── README.md

├── knn_heart_model.pkl

├── heart_scaler.pkl

├── heart_columns.pkl

└── frontend/

```

---

## ⚙️ Installation

```bash

git clone https://github.com/prithvicoder1/HeartVision-AI.git

cd HeartVision-AI

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

```

---

## 🐳 Docker

```bash

docker build -t heartvision-ai .

docker run -p 8501:8501 heartvision-ai

```

---

## 🧠 Input Features

- Age

- Sex

- Chest Pain Type

- Resting Blood Pressure

- Cholesterol

- Fasting Blood Sugar

- Resting ECG

- Maximum Heart Rate

- Exercise Angina

- Oldpeak

- ST Slope

---

## 📈 Prediction Output

- ✅ Low Risk

- ⚠️ High Risk

---

## 🚀 Future Improvements

- Probability score

- Explainable AI (SHAP)

- Multiple ML model comparison

- Authentication

- Patient report export (PDF)

---

## 🤝 Contributing

Pull requests and suggestions are welcome.

---

## ⚠️ Disclaimer

This project is for educational purposes only and is not a substitute for professional medical advice.

---

<p align="center">

Made with ❤️ by <b>Prithvi Vijay</b>

</p>

<p align="center">

⭐ If you like this project, consider starring the repository!

</p>
