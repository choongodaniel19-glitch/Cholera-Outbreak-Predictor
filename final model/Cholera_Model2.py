#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("final_combined_data.csv")

# 2. Clean and encode
df = df.dropna(subset=["TAVG_temperature", "Precipitation", "Reported cholera cases", "PopulationDensity"])
df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

le = LabelEncoder()
df["Country_Code"] = le.fit_transform(df["Country"])
joblib.dump(le, "label_encoder.pkl")

# 3. Create outbreak label
df["cases_per_100k"] = df["Reported cholera cases"] / df["PopulationDensity"]
df["Outbreak"] = (df["cases_per_100k"] > 1).astype(int)
if df["Outbreak"].nunique() < 2:
    df["Outbreak"] = (df["Reported cholera cases"] > 0).astype(int)

# 4. Create time features (seasonality)
df["Year_sin"] = np.sin(2 * np.pi * df["Year"] / 5)
df["Year_cos"] = np.cos(2 * np.pi * df["Year"] / 5)

# 5. Add derived features (new)
# 5a. Change in Population Density from previous year (per country)
df["PopDensity_change"] = df.groupby("Country")["PopulationDensity"].diff()

# 5b. Interaction feature between rainfall and temperature
df["Rain_x_Temp"] = df["TAVG_temperature"] * df["Precipitation"]

# 6. Select features
features = [
    "TAVG_temperature", "Precipitation", "Country_Code",
    "Year_sin", "Year_cos",
    "PopDensity_change", "Rain_x_Temp"
]

df = df.dropna(subset=features + ["Outbreak"])
X = df[features]
y = df["Outbreak"]

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Handle class imbalance
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 9. Train model
model = XGBClassifier(random_state=42, n_estimators=300, learning_rate=0.05)
model.fit(X_train_sm, y_train_sm)

# 10. Predict and evaluate
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

joblib.dump(model, "Cholera_Model2.pkl")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Feature importance
importances = model.get_booster().get_score(importance_type="gain")
pd.Series(importances).sort_values().plot.barh(figsize=(8,5))
plt.title("Feature Importance")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()


# In[18]:


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




