import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# Load model and encoders
@st.cache_resource
def load_model():
    return pickle.load(open("salary_model.pkl", "rb"))

model = load_model()

# Dropdown categories (based on LabelEncoder mapping from training)
education_options = [
    'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctorate',
    'High School', 'Professional Certification', 'Diploma'
]

job_title_options = [
    'Software Engineer', 'Data Analyst', 'Project Manager', 'Sales Manager',
    'Product Manager', 'Data Scientist', 'UX Designer', 'Marketing Manager',
    'HR Manager', 'Operations Manager'
    # ⬆️ Replace with actual top job titles from your dataset or full list
]

# Title
st.title("💼 Salary Prediction App")
st.markdown("Predict salary based on experience, education, and job title.")
st.markdown("---")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    education = st.selectbox("🎓 Education Level", education_options)

with col2:
    job_title = st.selectbox("💼 Job Title", job_title_options)

experience = st.slider("⌛ Years of Experience", 0.0, 40.0, 2.0, 0.5)

# Encode inputs manually (LabelEncoder simulation)
edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}

edu_encoded = edu_mapping.get(education, 0)
job_encoded = job_mapping.get(job_title, 0)

# Predict
if st.button("🎯 Predict Salary"):
    input_data = np.array([[experience, edu_encoded, job_encoded]])
    predicted_salary = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: **${predicted_salary:,.2f}**")

    # Visual: Prediction point on trend
    st.subheader("📈 Salary Prediction Visualization")
    x_vals = np.linspace(0, 40, 100)
    y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="Salary Trend", color='blue')
    ax.scatter(experience, predicted_salary, color='red', s=100, label="Your Prediction")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Years of Experience vs. Salary")
    ax.legend()
    st.pyplot(fig)

# Model metrics
with st.expander("📋 Model Performance (Random Forest)"):
    st.markdown("""
    | Metric   | Value     |
    |----------|-----------|
    | MAE      | 9,873.85  |
    | MSE      | 193,135,734.42 |
    | RMSE     | 13,897.33 |
    | R² Score | 0.9194    |
    """)

# About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app was created for the **2025 Summer Internship Program (AI Track)** at **DIGIPEX Solutions LLC**.
    It predicts salaries using a machine learning model trained on a dataset including:
    - Education level
    - Job title
    - Years of experience

    Model: **Random Forest Regressor**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size: 14px;'>🚀 Created by <a href='https://github.com/Malik9544' target='_blank'>MUHAMMAD_MUDASIR</a> | <a href='https://github.com/Malik9544/Salary_prediction_Model' target='_blank'>Project Repository</a></p>",
    unsafe_allow_html=True
)
