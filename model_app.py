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

# Manually define top 10 features
top_features = [
    "G2", "absences", "reason_home", "age", "G1",
    "famrel", "reason_course", "health", "goout", "schoolsup_no"
]

st.markdown("---")
st.subheader("🔍 Enter student data to predict G3 score")

# Input form
user_input = {}

for feature in top_features:
    if feature == "G2":
        user_input[feature] = st.slider("Previous Grade (G2)", 0, 20, 10)
    elif feature == "G1":
        user_input[feature] = st.slider("First Period Grade (G1)", 0, 20, 10)
    elif feature == "absences":
        user_input[feature] = st.slider("Number of Absences", 0, 100, 5)
    elif feature == "age":
        user_input[feature] = st.slider("Age", 15, 22, 17)
    elif feature == "famrel":
        user_input[feature] = st.slider("Family Relationship (1 = very bad, 5 = excellent)", 1, 5, 3)
    elif feature == "health":
        user_input[feature] = st.slider("Health Status (1 = very bad, 5 = excellent)", 1, 5, 3)
    elif feature == "goout":
        user_input[feature] = st.slider("Going Out with Friends (1 = very low, 5 = very high)", 1, 5, 3)
    elif feature == "reason_home":
        user_input[feature] = 1 if st.selectbox("Reason for School: Home?", ["No", "Yes"]) == "Yes" else 0
    elif feature == "reason_course":
        user_input[feature] = 1 if st.selectbox("Reason for School: Course?", ["No", "Yes"]) == "Yes" else 0
    elif feature == "schoolsup_no":
        user_input[feature] = 1 if st.selectbox("School Support: No?", ["No", "Yes"]) == "Yes" else 0
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Add missing columns (from training set)
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]  # Ensure same order

# Predict
if st.button("Predict G3 Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"📘 Predicted G3 Score: {prediction:.2f}")
