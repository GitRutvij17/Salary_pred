# app.py (fixed & defensive loader)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import plotly.express as px
import traceback
import re

st.set_page_config(page_title="Data Science Salary Predictor", page_icon="💰", layout="wide")

# -------------------------
# Robust model loader
# -------------------------
@st.cache_resource
def load_model(model_path="best_salary_model_gradient_boosting.pkl"):
    """Try joblib then pickle. Return (model, error_message_or_None)."""
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    joblib_tb = None
    pickle_tb = None

    # Try joblib
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception:
        joblib_tb = traceback.format_exc()

    # Try pickle
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception:
        pickle_tb = traceback.format_exc()

    # Both failed -> return combined tracebacks
    combined = "Joblib loader traceback:\n" + joblib_tb + "\n\nPickle loader traceback:\n" + pickle_tb
    return None, combined

# -------------------------
# Load dataset & features
# -------------------------
@st.cache_data
def load_data(path="clean_data.csv"):
    if not os.path.exists(path):
        return None, f"Dataset not found: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception:
        return None, traceback.format_exc()

@st.cache_data
def load_features(path="model_features.pkl"):
    if not os.path.exists(path):
        return None, None
    # try joblib then pickle
    try:
        feats = joblib.load(path)
        return feats, None
    except Exception:
        try:
            with open(path, "rb") as f:
                feats = pickle.load(f)
            return feats, None
        except Exception:
            return None, traceback.format_exc()

# -------------------------
# Get objects
# -------------------------
model, model_err = load_model()
data, data_err = load_data()
features, features_err = load_features()

# Sidebar status & helpful diagnostics
st.sidebar.title("Status")
if model is not None:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.error("❌ Model not loaded")
    if model_err:
        # show a short excerpt of the error and helpful hints
        st.sidebar.markdown("**Loader error (truncated):**")
        st.sidebar.code(model_err[:1000] + ("\n... (truncated)" if len(model_err) > 1000 else ""))
        # detect missing modules like "No module named 'loss'"
        missing = re.findall(r"No module named ['\"]([^'\"]+)['\"]", model_err)
        if missing:
            st.sidebar.warning(f"Missing module(s) detected: {', '.join(missing)}")
            st.sidebar.markdown("""
            **Fixes:**  
            - If the missing name is a standard package, install it: `pip install <package>`  
            - If it's a custom module (e.g. `loss`), make sure that `.py` file is available in your project folder or install the package that provides it.
            - If the model was saved using a custom class/function from your training script, re-save the model after moving that class into a module available at load time.
            """)
        # common numpy mismatch hint
        if "numpy._core" in model_err or "No module named numpy._core" in model_err:
            st.sidebar.info("Try reinstalling a compatible numpy version, e.g.: `pip install --force-reinstall numpy==1.26.4`")

if data is not None:
    st.sidebar.success(f"✅ Dataset loaded ({len(data):,} rows)")
else:
    st.sidebar.error("⚠️ Dataset not loaded")
    if data_err:
        st.sidebar.code(data_err[:800] + ("\n... (truncated)" if len(data_err) > 800 else ""))

# -------------------------
# UI pages
# -------------------------
page = st.sidebar.radio("Go to:", ["Home", "Predict Salary", "Dataset Overview"])

if page == "Home":
    st.title("💼 Data Science Salary Predictor")
    st.write("Model file: `best_salary_model_gradient_boosting.pkl`")
    st.write("Dataset file: `clean_data.csv`")
    st.write("Features file (optional): `model_features.pkl`")

elif page == "Predict Salary":
    st.header("Predict your salary")

    if model is None or data is None:
        st.error("Model or dataset not loaded. Check sidebar for diagnostics.")
    else:
        # inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
            company_size = st.selectbox("Company Size", ["S", "M", "L"])
        with col2:
            job_title = st.selectbox("Job Title", sorted(data['job_title'].unique()))
            employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
        with col3:
            work_year = st.slider("Work Year", int(data['work_year'].min()), int(data['work_year'].max()), int(data['work_year'].max()))
            employee_country = st.selectbox("Employee Country", sorted(data['employee_residence'].unique()))

        company_country = st.selectbox("Company Location", sorted(data['company_location'].unique()))
        remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 100, step=25)

        if st.button("Predict"):
            # simple preprocessing similar to your original function (you can expand)
            input_df = pd.DataFrame({
                'experience_level': [experience],
                'employment_type': [employment_type],
                'job_title': [job_title],
                'remote_ratio': [remote_ratio],
                'company_size': [company_size],
                'work_year': [work_year],
                'employee_residence': [employee_country],
                'company_location': [company_country]
            })

            # If features list exists, align columns and fill missing with 0
            if features is not None:
                try:
                    # features may be e.g. a list of column names
                    for c in features:
                        if c not in input_df.columns:
                            input_df[c] = 0
                    input_df = input_df[features]
                except Exception as e:
                    st.warning("Could not align features automatically: " + str(e))

            # Predict and show
            try:
                pred = model.predict(input_df)
                salary = float(pred[0])
                st.success(f"Predicted annual salary: ${salary:,.2f}")
                st.metric("Monthly (approx.)", f"${salary/12:,.2f}")
            except Exception as e:
                st.error("Prediction failed. See details below.")
                st.exception(e)

elif page == "Dataset Overview":
    st.header("Dataset overview")
    if data is None:
        st.error("Dataset not loaded.")
    else:
        st.dataframe(data.head())
        st.write(f"Rows: {len(data):,}")
        if "salary_in_usd" in data.columns:
            st.metric("Average salary", f"${data['salary_in_usd'].mean():,.0f}")
            fig = px.histogram(data, x="salary_in_usd", nbins=50, title="Salary distribution")
            st.plotly_chart(fig, use_container_width=True)
