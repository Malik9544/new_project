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

# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")

df = load_dataset()

# Encoded mappings
education_options = df["Education Level"].dropna().unique().tolist()
job_title_options = df["Job Title"].dropna().unique().tolist()

edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}

# Sidebar - user input
with st.sidebar:
    st.title("User Input")
    st.subheader(" Education & Career")
    education = st.selectbox("Select Your Education Level", education_options)
    job_title = st.selectbox("Select Your Job Title", job_title_options)

    st.subheader(" Experience")
    experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)

# Tabs
tabs = st.tabs([" Predict Salary", " Model Performance", " Logs", " Data Visualizations", "ℹ About"])

with tabs[0]:
    st.markdown("###  **Prediction Result**")
    if st.button(" Predict Salary Now"):
        edu_encoded = edu_mapping.get(education, 0)
        job_encoded = job_mapping.get(job_title, 0)
        input_data = np.array([[experience, edu_encoded, job_encoded]])

        with st.spinner("Generating prediction..."):
            time.sleep(1.2)
            predicted_salary = model.predict(input_data)[0]

        col1, col2 = st.columns(2)
        col1.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")
        col2.metric(label="Years of Experience", value=f"{experience} years")

        # Save log
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
        st.success(" Prediction logged successfully!")

        st.download_button("⬇ Download Prediction Result as CSV", data=pd.DataFrame([log_entry]).to_csv(index=False), file_name="prediction_result.csv", mime="text/csv")

        # Plotly chart for salary trend
        st.markdown("###  Predicted Salary Trend")
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

with tabs[1]:
    st.markdown("###  Model Performance (Random Forest Regressor)")
    st.markdown("""
    | Metric   | Value     |
    |----------|-----------|
    | MAE      | 9,873.85  |
    | MSE      | 193,135,734.42 |
    | RMSE     | 13,897.33 |
    | R² Score | 0.9194    |
    """)

with tabs[2]:
    st.markdown("### 🧾 View Prediction Logs")
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("⬇️ Download Full Log as CSV", data=log_data.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet.")

with tabs[3]:
    st.markdown("###  Job Title–Wise Average Salary")
    job_avg_salary = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).reset_index()
    fig1 = px.bar(job_avg_salary, x="Job Title", y="Salary", title="Average Salary by Job Title", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("###  Education Level Salary Impact")
    fig2 = px.box(df, x="Education Level", y="Salary", title="Salary by Education Level", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("###  Experience vs Salary (All Data)")
    fig3 = px.scatter(df, x="Years of Experience", y="Salary", color="Education Level", title="Experience vs Salary", template="simple_white")
    st.plotly_chart(fig3, use_container_width=True)

with tabs[4]:
    st.markdown("### ℹ️ About This App")
    st.info("""
    This salary prediction app was built as part of the **2025 Summer Internship at DIGIPEX Solutions LLC**.

    -  Built with Python, Streamlit, and Scikit-learn  
    -  Predicts salary using Education Level, Job Title, and Years of Experience  
    -  Model used: Random Forest Regressor  
    -  Visualization shows salary trend across experience years
    """)

# Footer
st.markdown("""
<hr style="border:1px solid #ccc;">
<p style='text-align:center; font-size: 14px;'>
 Created = by <a href='https://github.com/Malik9544' target='_blank'>MUHAMMAD_MUDASIR</a> |
<a href='https://salarypredictionmodel-8tfx9nxanp55wrqoxgbgm3.streamlit.app/' target='_blank'>Live App</a> |
<a href='https://github.com/Malik9544/Salary_prediction_Model' target='_blank'>GitHub Repo</a>
</p>
""", unsafe_allow_html=True)
