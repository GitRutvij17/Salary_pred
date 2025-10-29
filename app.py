# app.py

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DS Salary Predictor", page_icon="💰")

st.title("💰 Data Science Salary Predictor")

# -------------------------------
# LOAD MODEL AND FEATURE NAMES
# -------------------------------
@st.cache_data
def load_model():
    model = joblib.load("best_salary_model_random_forest.pkl")  # Your trained RF model
    feature_names = joblib.load("feature_names.pkl")  # Columns used in training
    return model, feature_names

rf_model, feature_names = load_model()

# -------------------------------
# USER INPUT
# -------------------------------
st.sidebar.header("Input Features")

# Dictionaries for encoding
experience_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
company_size_map = {'S': 1, 'M': 2, 'L': 3}
employment_map = {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4}

experience_level = st.sidebar.selectbox("Experience Level", list(experience_map.keys()))
company_size = st.sidebar.selectbox("Company Size", list(company_size_map.keys()))
employment_type = st.sidebar.selectbox("Employment Type", list(employment_map.keys()))
remote_ratio = st.sidebar.slider("Remote Ratio (%)", 0, 100, 0, step=10)
work_year = st.sidebar.number_input("Work Year", min_value=2020, max_value=2030, value=2023)
job_title = st.sidebar.text_input("Job Title (e.g., Data Scientist)")
employee_residence = st.sidebar.text_input("Employee Country (2-letter code, e.g., US)")
company_location = st.sidebar.text_input("Company Country (2-letter code, e.g., US)")

# -------------------------------
# PREPARE INPUT DATAFRAME
# -------------------------------
X_input = pd.DataFrame(0, index=[0], columns=feature_names)

# Encode basic features
X_input.loc[0, 'experience_encoded'] = experience_map[experience_level]
X_input.loc[0, 'company_size_encoded'] = company_size_map[company_size]
X_input.loc[0, 'employment_type_encoded'] = employment_map[employment_type]
X_input.loc[0, 'remote_ratio'] = remote_ratio
X_input.loc[0, 'work_year'] = work_year
X_input.loc[0, 'same_location'] = int(employee_residence == company_location)
X_input.loc[0, 'is_remote'] = int(remote_ratio == 100)
X_input.loc[0, 'is_hybrid'] = int(remote_ratio == 50)

# One-hot for job_title
job_col = f"job_{job_title}"
if job_col in X_input.columns:
    X_input.loc[0, job_col] = 1

# One-hot for employee_residence
emp_col = f"emp_country_{employee_residence}"
if emp_col in X_input.columns:
    X_input.loc[0, emp_col] = 1

# One-hot for company_location
comp_col = f"comp_country_{company_location}"
if comp_col in X_input.columns:
    X_input.loc[0, comp_col] = 1

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Salary"):
    try:
        salary_pred = rf_model.predict(X_input)[0]
        st.success(f"💵 Predicted Salary (USD): ${salary_pred:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
