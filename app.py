import streamlit as st
import numpy as np
import pickle
import json


# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load model + symptoms (cached)
@st.cache_resource
def load_model_and_symptoms():
    # Load ensemble model
    with open("models/disease_model_ensemble.pkl", "rb") as f:
        ensemble = pickle.load(f)

    rf = ensemble["random_forest"]
    lr = ensemble["logistic_regression"]
    rf_weight = ensemble["rf_weight"]
    lr_weight = ensemble["lr_weight"]
    classes = ensemble["classes"]

    # Load symptoms list
    with open("models/symptoms.json", "r") as f:
        symptoms = json.load(f)

    return rf, lr, rf_weight, lr_weight, classes, symptoms


rf, lr, rf_weight, lr_weight, classes, symptoms = load_model_and_symptoms()

# UI
st.title("🏥 Disease Prediction System")
st.write("### Enter the symptoms you're experiencing")

selected_symptoms = st.multiselect(
    "Select symptoms:",
    options=symptoms,
    default=[],
    help="Start typing to search for symptoms"
)

# Create input vector
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Medical safety filter
def is_valid_prediction(disease, selected):
    selected = set(selected)

    disease = disease.lower()

    if disease == "heart attack":
        return ("chest_pain" in selected) or ("shortness_of_breath" in selected)

    if disease in ["varicose veins"]:
        return False

    return True


# Prediction

if st.button("Predict Disease"):
    if len(selected_symptoms) < 3:
        st.warning("Please select at least 3 symptoms for better accuracy.")
    else:
        # Ensemble probabilities
        rf_probs = rf.predict_proba([input_vector])[0]
        lr_probs = lr.predict_proba([input_vector])[0]

        final_probs = (rf_weight * rf_probs) + (lr_weight * lr_probs)

        # Sort predictions
        sorted_idx = np.argsort(final_probs)[::-1]

        results = []
        for i in sorted_idx:
            disease = classes[i]
            prob = final_probs[i] * 100

            if is_valid_prediction(disease, selected_symptoms):
                prob = min(prob, 90)  # cap confidence
                results.append((disease, prob))

            if len(results) == 5:
                break

        # Display results
        st.success("### Possible Conditions (AI-based, Not a Diagnosis)")

        for i, (disease, prob) in enumerate(results, 1):
            st.write(f"**{i}. {disease}** – {prob:.1f}% confidence")

        # Key symptoms (from Random Forest)
        st.write("### Key symptoms contributing to this prediction:")
        important_indices = np.argsort(rf.feature_importances_)[::-1]

        shown = 0
        for idx in important_indices:
            if input_vector[idx] == 1:
                st.write(f"- {symptoms[idx].replace('_', ' ').title()}")
                shown += 1
            if shown == 5:
                break

        st.warning(
            "This system is for educational purposes only and must not be used as a substitute for professional medical advice."
        )

# Footer
st.markdown("---")
st.markdown("""
### About this System
This disease prediction system uses an ensemble of machine learning models 
(Random Forest + Logistic Regression) to suggest **possible conditions** based on symptoms.

The output is **not a medical diagnosis**.
""")
