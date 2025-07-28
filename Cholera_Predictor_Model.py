#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("Cholera_Model2.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset for reference (used for country history)
df = pd.read_csv("final_combined_data.csv")
df = df.dropna(subset=["Country", "TAVG_temperature", "Precipitation", "Reported cholera cases", "PopulationDensity"])
df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

# Prepare country list
country_list = sorted(df["Country"].unique())

# App UI
st.title("Cholera Outbreak Predictor")

# User input
country = st.selectbox("Country", country_list)
year = st.number_input("Year", min_value=2000, max_value=2030, step=1)
temperature = st.number_input("Average Temperature (°C)")
rainfall = st.number_input("Precipitation (mm)")

# Predict button
if st.button("Predict"):
    # Encode country
    try:
        country_code = le.transform([country])[0]
    except ValueError:
        st.error("Country not recognized.")
        st.stop()

    # Cyclical features
    year_sin = np.sin(2 * np.pi * year / 5)
    year_cos = np.cos(2 * np.pi * year / 5)

    # Estimate PopDensity_change using previous year data
    country_df = df[df["Country"] == country]
    last_year_data = country_df[country_df["Year"] == year - 1]

    if not last_year_data.empty:
        current_year_data = country_df[country_df["Year"] == year]
        if not current_year_data.empty:
            change = current_year_data.iloc[0]["PopulationDensity"] - last_year_data.iloc[0]["PopulationDensity"]
        else:
            change = last_year_data.iloc[0]["PopulationDensity"] * 0.02
    else:
        change = 0

    # Interaction feature
    rain_x_temp = temperature * rainfall

    # Assemble features
    input_data = pd.DataFrame([{
        "TAVG_temperature": temperature,
        "Precipitation": rainfall,
        "Country_Code": country_code,
        "Year_sin": year_sin,
        "Year_cos": year_cos,
        "PopDensity_change": change,
        "Rain_x_Temp": rain_x_temp
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Result
    if prediction == 1:
        if prob > 0.8:
            st.warning("🚨 Very High Risk of Outbreak! Take Immediate Precautions.")
        elif prob > 0.6:
            st.warning("⚠️ Moderate Risk. Monitor and prepare resources.")
        else:
            st.info("⚠️ Slight Risk. Stay alert and monitor conditions.")
    else:
        st.success(f"✅ No Outbreak Likely (Confidence: {prob:.2f})")


# In[ ]:




