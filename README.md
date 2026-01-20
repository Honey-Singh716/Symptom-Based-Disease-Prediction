# 🏥 Symptom-Based Disease Prediction System

An AI-powered **symptom-based risk suggestion system** built using machine learning.
This project demonstrates how ensemble models behave when predicting possible
medical conditions from user-reported symptoms.

⚠️ **Disclaimer:**  
This system is for **educational purposes only** and **must not be used for medical diagnosis**.

---

## 🚀 Features

- Interactive **Streamlit web application**
- User-friendly symptom selection
- **Ensemble model** (Random Forest + Logistic Regression)
- Top-5 possible condition suggestions
- Confidence capping to prevent overconfidence
- Handles **sparse real-world symptom input**
- Medical safety disclaimers included

---

## 🧠 Machine Learning Approach

- **Random Forest**  
  - Captures non-linear relationships between symptoms

- **Logistic Regression**  
  - Provides stable probability estimates

- **Soft Voting Ensemble**
  - Final probability = weighted average of both models

---

## 📊 Dataset

- Source: Public Kaggle Disease Prediction Dataset
- Format: Binary symptom indicators
- Nature: Synthetic / rule-based

⚠️ Due to privacy and ethical constraints, real-world patient-level medical data
is not publicly available. This dataset is used to demonstrate ML behavior
and limitations.

---

## 🧪 Model Evaluation

- Training vs Validation accuracy comparison
- Learning curve analysis
- Top-K accuracy used instead of single-label accuracy
- Manual testing using realistic symptom combinations

Expected real-world plausibility: **~60–70%**

---

## 🖥️ Application Preview

Screenshots are available in the `screenshots/` folder.

---

## 📁 Project Structure

```
disease-prediction/
├── app.py                    # Streamlit web app
├── models/
│   ├── disease_model_ensemble.pkl  # Trained ensemble model
│   └── symptoms.json           # Symptom list
├── screenshots/              # App screenshots
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # License file
```



---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Disease-Prediction-System.git
cd Disease-Prediction-System




### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---
