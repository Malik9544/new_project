# 💼 Salary Prediction Using Traditional ML Techniques

This project is part of the **2025 Summer Internship Program – AI Track** at **DIGIPEX Solutions LLC**. The goal is to build a regression model that predicts an employee's salary based on years of experience using traditional machine learning techniques, and to deploy this model using **Streamlit** on **Render**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Phases](#project-phases)
- [Model Evaluation](#model-evaluation)
- [Streamlit App](#streamlit-app)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🧠 Overview

This project involves:
- Exploring and preprocessing employee salary data
- Training multiple regression models (Linear Regression and Random Forest)
- Evaluating performance using standard regression metrics
- Deploying the best model in an interactive Streamlit web application

---

## 📊 Dataset

- Dataset Name: **Employee Data for Salary Prediction**
- Source: [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/employee-data-for-salary-prediction)
- Features Used: 
  - `YearsExperience` (float)
  - `Salary` (float - target variable)

---

## 🔍 Project Phases

1. **Data Preprocessing**: Cleaned dataset, handled datatypes
2. **Feature Engineering**: No derived features needed
3. **Model Building**: Linear Regression & Random Forest Regressor
4. **Evaluation**: Metrics compared across models
5. **Deployment**: Streamlit app deployed on Render

---

## 📈 Model Evaluation

| Metric   | Linear Regression | Random Forest |
| -------- | ----------------- | ------------- |
| MAE      | ✅ **6286.45**     | 6872.01       |
| MSE      | ✅ **49.8M**       | 63.7M         |
| RMSE     | ✅ **7059.04**     | 7982.55       |
| R² Score | ✅ **0.9024**      | 0.8753        |

> ✅ Final Model Selected: **Linear Regression**

---

## 🌐 Streamlit App

🔗 **Live Demo:** [Click here to try the app][(🔗 **Live Demo:** [Click here to try the app](https://salarypredictionmodel-8tfx9nxanp55wrqoxgbgm3.streamlit.app)


Features:
- Input: Years of Experience (slider)
- Output: Predicted Salary
- Real-time prediction

---

## 🧰 Tech Stack

- **Python**
- **Pandas, NumPy**
- **scikit-learn**
- **Matplotlib, Seaborn**
- **Streamlit**
- **Render** (for deployment)

---

## 💻 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/salary-prediction-ml.git
   cd salary-prediction-ml
