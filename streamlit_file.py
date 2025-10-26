import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load model & scaler
# --------------------------
rf_model = joblib.load("rf_brainstroke_model.pkl")
scaler = joblib.load("scaler_brainstroke.pkl")

TRAIN_COLUMNS = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                 'bmi', 'gender_Male', 'ever_married_Yes', 'work_type_Private',
                 'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
                 'smoking_status_formerly smoked', 'smoking_status_never smoked',
                 'smoking_status_smokes']

# --------------------------
# Page Layout
# --------------------------
st.set_page_config(page_title="🧠 Brain Stroke Predictor", layout="wide")
st.markdown("""
<div style='text-align:center; background: linear-gradient(to right, #6a11cb, #2575fc); 
            padding:25px; border-radius:15px; color:white;'>
    <h1>🧠 Brain Stroke Predictor</h1>
    <p>Interactive & visually stunning prediction</p>
</div>
""", unsafe_allow_html=True)
st.write("")

# --------------------------
# Input Form in Columns
# --------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("🧓 Age", 0, 120, 50)
    hypertension = st.radio("⚕️ Hypertension", [0,1])
    heart_disease = st.radio("❤️ Heart Disease", [0,1])
    avg_glucose = st.number_input("🩸 Avg Glucose Level", 0.0, 500.0, 100.0)

with col2:
    bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0)
    gender = st.radio("👤 Gender", ["Male","Female"])
    married = st.radio("💍 Ever Married?", ["Yes","No"])
    work_type = st.selectbox("💼 Work Type", ["Private","Self-employed","Children"])

with col3:
    residence = st.radio("🏠 Residence", ["Urban","Rural"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["formerly smoked","never smoked","smokes"])
    predict_btn = st.button("Predict Stroke Risk")

# --------------------------
# Preprocess Input
# --------------------------
def preprocess_input():
    input_dict = dict.fromkeys(TRAIN_COLUMNS, 0)
    input_dict['age'] = age
    input_dict['hypertension'] = hypertension
    input_dict['heart_disease'] = heart_disease
    input_dict['avg_glucose_level'] = avg_glucose
    input_dict['bmi'] = bmi
    input_dict['gender_Male'] = 1 if gender.lower()=="male" else 0
    input_dict['ever_married_Yes'] = 1 if married.lower()=="yes" else 0
    
    if work_type.lower()=="private":
        input_dict['work_type_Private'] = 1
    elif work_type.lower()=="self-employed":
        input_dict['work_type_Self-employed'] = 1
    elif work_type.lower()=="children":
        input_dict['work_type_children'] = 1
        
    input_dict['Residence_type_Urban'] = 1 if residence.lower()=="urban" else 0
    
    if smoking_status.lower()=="formerly smoked":
        input_dict['smoking_status_formerly smoked'] = 1
    elif smoking_status.lower()=="never smoked":
        input_dict['smoking_status_never smoked'] = 1
    elif smoking_status.lower()=="smokes":
        input_dict['smoking_status_smokes'] = 1
        
    return pd.DataFrame([input_dict], columns=TRAIN_COLUMNS)

# --------------------------
# Prediction & Display
# --------------------------
if predict_btn:
    input_df = preprocess_input()
    input_scaled = scaler.transform(input_df)
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][prediction]*100

    # Fixed card size for sleek look
    width = 350
    height = 150

    if prediction == 1:
        color = f"linear-gradient(to right, #FF4B4B, #FF7777)"
        result_text = "High Risk ⚠️"
    else:
        color = f"linear-gradient(to right, #4CAF50, #80E27E)"
        result_text = "Low Risk ✅"

    st.markdown(f"""
    <div style='width:{width}px; height:{height}px; margin:auto; background:{color}; 
                border-radius:20px; text-align:center; padding:20px; color:white; 
                box-shadow: 0 0 20px rgba(0,0,0,0.3);'>
        <h2 style='font-size:28px'>{result_text}</h2>
        <h3 style='font-size:22px'>Confidence: {probability:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Animated risk meter
    st.subheader("Risk Meter")
    st.progress(int(probability))

    # Probability bar chart
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": ["No Stroke","Stroke"],
        "Probability": [rf_model.predict_proba(input_scaled)[0][0]*100,
                        rf_model.predict_proba(input_scaled)[0][1]*100]
    })
    st.bar_chart(prob_df.set_index("Class"))
