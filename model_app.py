import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🎓 Student Final Grade (G3) Predictor")

# Load data
file_path = "modified.xls"
df = pd.read_excel(file_path, engine='xlrd')

# Preprocess
X = df.drop(columns=["G3"])
y = df["G3"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_n = 10
top_features = importance_df['Feature'].head(top_n).tolist()

st.subheader(f"Top {top_n} Important Features:")
st.write(top_features)

st.markdown("---")
st.subheader("🔍 Enter student data to predict G3 score")

# UI inputs
user_input = {}

# Reason (merged reason_xx columns)
if any(f.startswith("reason_") for f in top_features):
    reason_options = [col.replace("reason_", "") for col in X.columns if col.startswith("reason_")]
    selected_reason = st.selectbox("Reason for choosing this school", reason_options)
    for reason in reason_options:
        col_name = f"reason_{reason}"
        user_input[col_name] = 1 if selected_reason == reason else 0

# School support (merged schoolsup_yes/no)
if any(f.startswith("schoolsup_") for f in top_features):
    support = st.radio("Extra educational support (schoolsup)", ["yes", "no"])
    for opt in ["yes", "no"]:
        col_name = f"schoolsup_{opt}"
        user_input[col_name] = 1 if support == opt else 0

# Remaining numerical features
for feature in top_features:
    if feature.startswith("reason_") or feature.startswith("schoolsup_"):
        continue
    elif feature == "age":
        user_input[feature] = st.slider("Age", min_value=15, max_value=22, value=17)
    elif feature in ["G1", "G2"]:
        user_input[feature] = st.slider(feature, min_value=0, max_value=20, value=10)
    elif feature == "absences":
        user_input[feature] = st.slider("Absences", min_value=0, max_value=100, value=5)
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0)

# Build input DataFrame
input_df = pd.DataFrame([user_input])

# Add missing dummy columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]  # Align column order

# Predict
if st.button("Predict G3 Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"📘 Predicted Final Grade (G3): {prediction:.2f}")
