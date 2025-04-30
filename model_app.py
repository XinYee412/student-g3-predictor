import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Title
st.title("🎓 Student G3 Grade Predictor")

# Load dataset and preprocess
@st.cache_data
def load_data():
    df = pd.read_excel("modified.xls", engine="xlrd")
    df = pd.get_dummies(df.drop(columns=["G3"]))
    return df

# Load data and train model
df = load_data()
target = pd.read_excel("modified.xls", engine="xlrd")["G3"]

# You can manually select the top 10 features from earlier analysis
top_features = [
    'G1', 'G2', 'failures', 'studytime', 'absences',
    'goout', 'health', 'Walc', 'Dalc', 'schoolsup_yes'
]

X = df[top_features]
y = target

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# User input
st.subheader("📋 Enter Student Data")

input_data = {}
for feature in top_features:
    if feature in ['schoolsup_yes']:
        input_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0)

# Predict button
if st.button("Predict G3 Score"):
    input_array = np.array([list(input_data.values())])
    prediction = model.predict(input_array)[0]
    st.success(f"🎯 Predicted G3 Score: {prediction:.2f}")
