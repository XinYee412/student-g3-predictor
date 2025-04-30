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

# Input form with sliders and selectboxes
user_input = {}
for feature in top_features:
    label = feature.replace("_", " ").capitalize()

    if feature.startswith('sex_'):
        user_input[feature] = st.selectbox("Gender", ["M", "F"]) == "M"

    elif "famrel" in feature:
        user_input[feature] = st.slider("Family Relationship Quality (1 = very bad, 5 = excellent)", 1, 5, 3)

    elif "studytime" in feature:
        user_input[feature] = st.slider("Weekly Study Time (1 = <2hrs, 4 = >10hrs)", 1, 4, 2)

    elif "failures" in feature:
        user_input[feature] = st.slider("Number of Past Class Failures", 0, 4, 0)

    elif "absences" in feature:
        user_input[feature] = st.slider("Absences", 0, 50, 5)

    elif df[feature].nunique() <= 10:
        user_input[feature] = st.slider(f"{label}", 0, 20, 10)

    else:
        user_input[feature] = st.slider(f"{label}", 0, 100, 0)

# Prepare input
input_df = pd.DataFrame([user_input])

# Fill missing dummy columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]  # Match training column order

# Predict
if st.button("Predict G3 Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"📘 Predicted G3 Score: {prediction:.2f}")
