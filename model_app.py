import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Final Grade (G3) Prediction System")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("student_grade_model.pkl")

model = load_model()

st.subheader("📝 Enter Student Information")

# Age
age = st.slider("Age (15 - 22 years)", 15, 22, 17, help="Student's age")

# First and second period grades
G1 = st.slider("First period grade (G1: 0 - 20)", 0, 20, 10, help="Grade from first term")
G2 = st.slider("Second period grade (G2: 0 - 20)", 0, 20, 10, help="Grade from second term")

# Absences
absences = st.slider("Number of school absences (0 - 93)", 0, 93, 5, help="Number of school absences")

# Family relationship quality
famrel = st.slider("Family relationship quality (1 - very bad to 5 - excellent)", 1, 5, 3,
                   help="Quality of relationships with family")

# Health status
health = st.slider("Current health status (1 - very bad to 5 - very good)", 1, 5, 3,
                   help="Student's current health condition")

# Going out with friends
goout = st.slider("Going out frequency (1 - very low to 5 - very high)", 1, 5, 3,
                  help="How often the student goes out with friends")

# Father's education level
Fedu = st.slider("Father's education (0 - none to 4 - higher education)", 0, 4, 2,
                 help="Father's highest education level")

# Reason for choosing the school
reason = st.selectbox("Reason for choosing this school", 
                      ["home", "reputation", "course", "other"],
                      help="Main reason the student chose this school")

# Extra educational support
schoolsup = st.radio("Extra educational support", ["yes", "no"],
                     help="Whether the student receives additional educational support")

# ---- Prepare data for model ----
user_input = {
    'age': age,
    'G1': G1,
    'G2': G2,
    'absences': absences,
    'famrel': famrel,
    'health': health,
    'goout': goout,
    'Fedu': Fedu,
    'reason_home': 1 if reason == "home" else 0,
    'reason_reputation': 1 if reason == "reputation" else 0,
    'reason_course': 1 if reason == "course" else 0,
    'reason_other': 1 if reason == "other" else 0,
    'schoolsup_yes': 1 if schoolsup == "yes" else 0,
    'schoolsup_no': 1 if schoolsup == "no" else 0
}

# Ensure input matches training feature order
X_all_columns = model.feature_names_in_
input_df = pd.DataFrame([user_input])
for col in X_all_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X_all_columns]

# Predict button
if st.button("🔍 Predict Final Grade (G3)"):
    prediction = model.predict(input_df)[0]
    st.success(f"📘 Predicted final grade (G3): **{prediction:.2f}** out of 20")
