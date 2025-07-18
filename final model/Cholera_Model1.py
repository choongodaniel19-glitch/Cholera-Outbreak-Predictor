#!/usr/bin/env python
# coding: utf-8

# In[25]:


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

# 5. Select only current-year features
features = [
    "TAVG_temperature", "Precipitation", "Country_Code",
    "Year_sin", "Year_cos"
]

X = df[features].dropna()
y = df.loc[X.index, "Outbreak"]

# 6. Split data (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Handle class imbalance
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 8. Train model
model = XGBClassifier(random_state=42, n_estimators=300, learning_rate=0.05)
model.fit(X_train_sm, y_train_sm)

# 9. Predict and evaluate
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

joblib.dump(model, "Cholera_Model.pkl")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 10. Feature importance plot
importances = model.get_booster().get_score(importance_type="gain")
pd.Series(importances).sort_values().plot.barh(figsize=(8,5))
plt.title("Feature Importance")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()


# In[26]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("Cholera_Model.pkl")
le = joblib.load("label_encoder.pkl")

# Load full dataset for reference (needed for feature engineering)
df = pd.read_csv("final_combined_data.csv")
df = df.dropna(subset=["Country", "TAVG_temperature", "Precipitation", "Reported cholera cases"])

# Prepare country list for user selection
country_list = sorted(df["Country"].unique())

# App title
st.title("Cholera Outbreak Predictor")

# User inputs
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
        st.error("Country not recognized by label encoder.")
        st.stop()

    # Compute cyclical year features
    year_sin = np.sin(2 * np.pi * year / 12)
    year_cos = np.cos(2 * np.pi * year / 12)

    # Create input DataFrame matching training features
    input_data = pd.DataFrame([{
        "TAVG_temperature": temperature,
        "Precipitation": rainfall,
        "Country_Code": country_code,
        "Year_sin": year_sin,
        "Year_cos": year_cos,
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Outbreak Likely (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ No Outbreak Likely (Confidence: {prob:.2f})")



# In[ ]:




