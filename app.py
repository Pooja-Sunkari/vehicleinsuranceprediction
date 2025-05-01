#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np

# Load model and preprocessing tools
model = joblib.load("rf_model_sampled.pkl")
scaler = joblib.load("scaler_sampled.pkl")
enc_gender = joblib.load("encoder_Gender_sampled.pkl")
enc_vehicle_age = joblib.load("encoder_Vehicle_Age_sampled.pkl")
enc_vehicle_damage = joblib.load("encoder_Vehicle_Damage_sampled.pkl")

st.title("🚗 Vehicle Insurance Interest Predictor")

st.markdown("Predict whether a customer is likely to be interested in purchasing vehicle insurance.")

# User input
gender = st.selectbox("Gender", enc_gender.classes_)
age = st.slider("Age", 20, 85, 30)
driving_license = st.selectbox("Driving License", ["No", "Yes"])
previously_insured = st.selectbox("Previously Insured", ["No", "Yes"])
vehicle_age = st.selectbox("Vehicle Age", enc_vehicle_age.classes_)
vehicle_damage = st.selectbox("Vehicle Damage", enc_vehicle_damage.classes_)
annual_premium = st.number_input("Annual Premium", value=30000)
region_code = st.number_input("Region Code", value=28)
policy_sales_channel = st.number_input("Policy Sales Channel", value=26)
vintage = st.slider("Customer Vintage (days)", 10, 300, 150)

if st.button("Predict"):
    # Encode categorical
    gender_encoded = enc_gender.transform([gender])[0]
    vehicle_age_encoded = enc_vehicle_age.transform([vehicle_age])[0]
    vehicle_damage_encoded = enc_vehicle_damage.transform([vehicle_damage])[0]
    driving_license_bin = 1 if driving_license == "Yes" else 0
    previously_insured_bin = 1 if previously_insured == "Yes" else 0

    # Create feature array
    features = np.array([[gender_encoded, age, driving_license_bin, previously_insured_bin,
                          vehicle_age_encoded, vehicle_damage_encoded, annual_premium,
                          region_code, policy_sales_channel, vintage]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][prediction]

    # Output
    if prediction == 1:
        st.success(f"✅ This customer is likely interested in insurance. (Confidence: {prob:.2f})")
    else:
        st.warning(f"⚠️ This customer is unlikely to be interested. (Confidence: {prob:.2f})")



# In[ ]:




