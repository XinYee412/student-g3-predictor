import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 用于加载模型

# 页面标题
st.title("🎓 Student Final Grade (G3) Predictor")

# 加载训练好的模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load("student_grade_model.pkl")  # 尝试加载模型
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure the model file is in the correct location.")
        return None  # 返回 None 表示模型加载失败

model = load_model()

# 如果模型加载失败，停止后续操作
if model is None:
    st.stop()

# 示例输入（你可以根据需要换成动态输入）
st.subheader("🔍 Enter student data to predict G3 score")

age = st.slider("Age", 15, 22, 17)
G1 = st.slider("G1 Score", 0, 20, 10)
G2 = st.slider("G2 Score", 0, 20, 10)
absences = st.slider("Absences", 0, 100, 5)
famrel = st.slider("Family Relationship Quality", 1, 5, 3)
health = st.slider("Health Status", 1, 5, 3)
goout = st.slider("Going Out with Friends", 1, 5, 3)
Fedu = st.slider("Father's Education Level", 0, 4, 2)

# reason 和 schoolsup 需要 One-hot 编码
reason = st.selectbox("Reason for choosing school", ["home", "course"])
schoolsup = st.radio("Extra Educational Support", ["yes", "no"])

# 构建输入特征字典
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
    'reason_course': 1 if reason == "course" else 0,
    'schoolsup_yes': 1 if schoolsup == "yes" else 0,
    'schoolsup_no': 1 if schoolsup == "no" else 0
}

# 转换成 DataFrame，并填补缺失列
X_all_columns = model.feature_names_in_  # 模型训练时的列名顺序
input_df = pd.DataFrame([user_input])
for col in X_all_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X_all_columns]

# 预测并显示结果
if st.button("Predict G3 Score"):
    pred = model.predict(input_df)[0]
    st.success(f"📘 Predicted Final Grade (G3): {pred:.2f}")
