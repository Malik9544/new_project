import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import datetime
import time

# Page config
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("salary_model.pkl", "rb"))

model = load_model()

# Encoded mappings
education_options = [
    'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctorate',
    'High School', 'Professional Certification', 'Diploma'
]

job_title_options = [
    'Software Engineer', 'Data Analyst', 'Project Manager', 'Sales Manager',
    'Product Manager', 'Data Scientist', 'UX Designer', 'Marketing Manager',
    'HR Manager', 'Operations Manager'
]

edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}

# Load dataset for analytics
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")  # Ensure this file exists in the project

df = load_dataset()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🎯 Predict Salary", "📈 Analytics", "📊 Model Info", "🧾 Logs", "ℹ️ About"])

# Sidebar input (only for prediction page)
if page == "🎯 Predict Salary":
    with st.sidebar:
        st.title("📥 User Input")
        st.subheader("🎓 Education & Career")
        education = st.selectbox("Select Your Education Level", education_options)
        job_title = st.selectbox("Select Your Job Title", job_title_options)

        st.subheader("⏳ Experience")
        experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)

        st.markdown("---")
        st.markdown("🔗 *App developed by [MUHAMMAD_MUDASIR](https://github.com/Malik9544)*")

# Top bar image and title
if page == "🎯 Predict Salary":
    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("<h1 style='text-align: center;'>💼 Interactive Salary Prediction App</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Predict your salary using Machine Learning trained on real-world employee data.</p>", unsafe_allow_html=True)
    with col2:
        st.image("https://avatars.githubusercontent.com/u/191113155?v=4", width=80)

    st.markdown("---")

    if st.button("🎯 Predict Salary Now"):
        edu_encoded = edu_mapping.get(education, 0)
        job_encoded = job_mapping.get(job_title, 0)
        input_data = np.array([[experience, edu_encoded, job_encoded]])

        with st.spinner("Generating prediction..."):
            time.sleep(1.2)
            predicted_salary = model.predict(input_data)[0]

        st.markdown("### 🧾 **Prediction Result**")
        col1, col2 = st.columns(2)
        col1.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")
        col2.metric(label="Years of Experience", value=f"{experience} years")

        log_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'education_level': education,
            'job_title': job_title,
            'years_experience': experience,
            'predicted_salary': predicted_salary
        }

        log_file = 'prediction_log.csv'
        if os.path.exists(log_file):
            log_df = pd.read_csv(log_file)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])

        log_df.to_csv(log_file, index=False)
        st.success("📝 Prediction logged successfully!")

        st.markdown("### 📈 Predicted Salary Trend")
        x_vals = np.linspace(0, 40, 100)
        y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Trend', line=dict(color='#00BFFF')))
        fig.add_trace(go.Scatter(x=[experience], y=[predicted_salary], mode='markers',
                                 name='Your Prediction', marker=dict(size=10, color='red')))
        fig.update_layout(title=f"Salary Trend for {job_title} with {education}",
                          xaxis_title='Years of Experience',
                          yaxis_title='Salary',
                          template='plotly_white',
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Analytics":
    st.header("📊 Salary Data Analytics")

    st.subheader("1. Average Salary by Job Title")
    avg_salary_job = df.groupby("Job Title")['Salary'].mean().sort_values(ascending=False)
    fig1 = px.bar(avg_salary_job, x=avg_salary_job.index, y=avg_salary_job.values,
                  labels={'x': 'Job Title', 'y': 'Average Salary'},
                  title="Average Salary by Job Title")
    fig1.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Salary Distribution by Education Level")
    fig2 = px.box(df, x='Education Level', y='Salary', color='Education Level',
                  title="Salary Distribution Across Education Levels")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Experience vs Salary (Scatter Plot)")
    fig3 = px.scatter(df, x='Years of Experience', y='Salary', color='Job Title',
                      title="Experience vs Salary by Job Title", opacity=0.6)
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

elif page == "📊 Model Info":
    st.header("📊 Model Performance (Random Forest Regressor)")
    st.markdown("""
    | Metric   | Value     |
    |----------|-----------|
    | MAE      | 9,873.85  |
    | MSE      | 193,135,734.42 |
    | RMSE     | 13,897.33 |
    | R² Score | 0.9194    |
    """)

elif page == "🧾 Logs":
    st.header("🧾 View Prediction Logs")
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("⬇️ Download Full Log as CSV", data=log_data.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet.")

elif page == "ℹ️ About":
    st.info("""
    This salary prediction app was built as part of the **2025 Summer Internship at DIGIPEX Solutions LLC**.

    - ✅ Built with Python, Streamlit, and Scikit-learn  
    - ✅ Predicts salary using Education Level, Job Title, and Years of Experience  
    - ✅ Random Forest Regressor for prediction  
    - 📈 Data-driven visual insights with Plotly  
    """)

# Feedback form in sidebar
with st.sidebar.expander("📬 Feedback / Contact"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")
    if st.button("📨 Send Feedback"):
        st.success("Thank you for your feedback!")

# Footer
st.markdown("""
<hr style="border:1px solid #ccc;">
<p style='text-align:center; font-size: 14px;'>
🚀 Created with ❤️ by <a href='https://github.com/Malik9544' target='_blank'>MUHAMMAD_MUDASIR</a> |
<a href='https://salarypredictionmodel-8tfx9nxanp55wrqoxgbgm3.streamlit.app/' target='_blank'>Live App</a> |
<a href='https://github.com/Malik9544/Salary_prediction_Model' target='_blank'>GitHub Repo</a>
</p>
""", unsafe_allow_html=True)
