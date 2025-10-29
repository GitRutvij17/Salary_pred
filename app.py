"""
Enhanced Data Science Salary Predictor
With INR conversion, analysis, and beautiful UI
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="DS Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .salary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL AND DATA
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_salary_model_random_forest.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, feature_names, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("clean_data.csv")
        return data, None
    except Exception as e:
        return None, str(e)

rf_model, feature_names, model_error = load_model()
data, data_error = load_data()

# USD to INR conversion rate (you can update this)
USD_TO_INR = 83.0  # Current approximate rate

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<h1 class="main-header">💰 Data Science Salary Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Predict salaries with AI-powered insights • USD & INR Support</p>", unsafe_allow_html=True)

# Status check
if rf_model is None:
    st.error(f"⚠️ Model loading failed: {model_error}")
    st.stop()

# -------------------------------
# SIDEBAR - NAVIGATION
# -------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["💰 Salary Predictor", "📊 Market Analysis", "ℹ️ About"])

# -------------------------------
# PAGE 1: SALARY PREDICTOR
# -------------------------------
if page == "💰 Salary Predictor":
    
    st.markdown("---")
    st.subheader("📝 Enter Your Details")
    
    # Dictionaries for encoding
    experience_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    company_size_map = {'S': 1, 'M': 2, 'L': 3}
    employment_map = {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4}
    
    # Input form in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Professional Info")
        experience_level = st.selectbox(
            "Experience Level",
            list(experience_map.keys()),
            format_func=lambda x: {'EN': '🟢 Entry Level', 'MI': '🟡 Mid Level', 
                                   'SE': '🟠 Senior', 'EX': '🔴 Executive'}[x]
        )
        
        employment_type = st.selectbox(
            "Employment Type",
            list(employment_map.keys()),
            format_func=lambda x: {'FT': '💼 Full-Time', 'PT': '⏰ Part-Time',
                                   'CT': '📋 Contract', 'FL': '🎯 Freelance'}[x]
        )
        
        job_title = st.text_input("Job Title", value="Data Scientist", 
                                  help="e.g., Data Scientist, ML Engineer, Data Analyst")
    
    with col2:
        st.markdown("#### 🏢 Company Info")
        company_size = st.selectbox(
            "Company Size",
            list(company_size_map.keys()),
            format_func=lambda x: {'S': '🏪 Small (<50)', 'M': '🏢 Medium (50-250)',
                                   'L': '🏛️ Large (>250)'}[x]
        )
        
        company_location = st.text_input("Company Location", value="US",
                                        help="2-letter country code (e.g., US, IN, GB)")
        
        work_year = st.number_input("Work Year", min_value=2020, max_value=2030, value=2024)
    
    with col3:
        st.markdown("#### 🌍 Location & Work")
        employee_residence = st.text_input("Your Location", value="IN",
                                          help="2-letter country code (e.g., US, IN, GB)")
        
        remote_ratio = st.select_slider(
            "Work Arrangement",
            options=[0, 50, 100],
            value=50,
            format_func=lambda x: {0: '🏢 On-site', 50: '🔄 Hybrid', 100: '🏠 Remote'}[x]
        )
        
        # Currency preference
        currency = st.radio("Preferred Currency", ["USD 💵", "INR ₹", "Both"], horizontal=True)
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔮 PREDICT MY SALARY", use_container_width=True):
        
        with st.spinner("🤖 AI is analyzing your profile..."):
            try:
                # Prepare input
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
                
                # One-hot encoding
                job_col = f"job_{job_title}"
                if job_col in X_input.columns:
                    X_input.loc[0, job_col] = 1
                
                emp_col = f"emp_country_{employee_residence}"
                if emp_col in X_input.columns:
                    X_input.loc[0, emp_col] = 1
                
                comp_col = f"comp_country_{company_location}"
                if comp_col in X_input.columns:
                    X_input.loc[0, comp_col] = 1
                
                # Predict
                salary_usd = rf_model.predict(X_input)[0]
                salary_inr = salary_usd * USD_TO_INR
                
                # Success animation
                st.balloons()
                
                # Display based on currency preference
                if currency == "USD 💵":
                    st.markdown(f"""
                    <div class="salary-card">
                        <h3 style="margin: 0;">Predicted Annual Salary</h3>
                        <h1 style="font-size: 3.5rem; margin: 1rem 0;">${salary_usd:,.0f}</h1>
                        <p style="font-size: 1.2rem; opacity: 0.9;">United States Dollars</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif currency == "INR ₹":
                    st.markdown(f"""
                    <div class="salary-card">
                        <h3 style="margin: 0;">Predicted Annual Salary</h3>
                        <h1 style="font-size: 3.5rem; margin: 1rem 0;">₹{salary_inr:,.0f}</h1>
                        <p style="font-size: 1.2rem; opacity: 0.9;">Indian Rupees</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:  # Both
                    col_usd, col_inr = st.columns(2)
                    with col_usd:
                        st.markdown(f"""
                        <div class="salary-card">
                            <h3>USD Amount</h3>
                            <h1 style="font-size: 2.5rem; margin: 1rem 0;">${salary_usd:,.0f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_inr:
                        st.markdown(f"""
                        <div class="salary-card">
                            <h3>INR Amount</h3>
                            <h1 style="font-size: 2.5rem; margin: 1rem 0;">₹{salary_inr:,.0f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # # Detailed Breakdown
                # st.subheader("💡 Salary Breakdown")
                
                # col_a, col_b, col_c, col_d = st.columns(4)
                
                # with col_a:
                #     st.markdown(f"""
                #     <div class="metric-card">
                #         <h4>💵 Monthly (USD)</h4>
                #         <h2>${salary_usd/12:,.0f}</h2>
                #         <p style="color: #666;">₹{salary_inr/12:,.0f}</p>
                #     </div>
                #     """, unsafe_allow_html=True)
                
                # with col_b:
                #     st.markdown(f"""
                #     <div class="metric-card">
                #         <h4>📅 Bi-Weekly (USD)</h4>
                #         <h2>${salary_usd/26:,.0f}</h2>
                #         <p style="color: #666;">₹{salary_inr/26:,.0f}</p>
                #     </div>
                #     """, unsafe_allow_html=True)
                
                # with col_c:
                #     st.markdown(f"""
                #     <div class="metric-card">
                #         <h4>⏰ Hourly (USD)</h4>
                #         <h2>${salary_usd/(52*40):,.0f}</h2>
                #         <p style="color: #666;">₹{salary_inr/(52*40):,.0f}</p>
                #     </div>
                #     """, unsafe_allow_html=True)
                
                # with col_d:
                #     if data is not None:
                #         percentile = (data['salary_in_usd'] < salary_usd).mean() * 100
                #         st.markdown(f"""
                #         <div class="metric-card">
                #             <h4>📊 Percentile</h4>
                #             <h2>{percentile:.0f}th</h2>
                #             <p style="color: #666;">In Dataset</p>
                #         </div>
                #         """, unsafe_allow_html=True)
                
                # st.markdown("<br>", unsafe_allow_html=True)
                
                # # Salary Range
                # st.markdown(f"""
                # <div class="info-box">
                #     <h4>💰 Typical Salary Range</h4>
                #     <p><strong>USD:</strong> ${salary_usd*0.85:,.0f} - ${salary_usd*1.15:,.0f}</p>
                #     <p><strong>INR:</strong> ₹{salary_inr*0.85:,.0f} - ₹{salary_inr*1.15:,.0f}</p>
                #     <p style="color: #666; margin-top: 0.5rem;">
                #     This represents a typical ±15% variation based on negotiation, benefits, 
                #     and company-specific factors.
                #     </p>
                # </div>
                # """, unsafe_allow_html=True)
                
                # # Market Insights
                # if data is not None:
                #     st.markdown("<br>", unsafe_allow_html=True)
                #     st.subheader("📈 Market Insights")
                    
                #     col_x, col_y = st.columns(2)
                    
                #     with col_x:
                #         # Similar profiles
                #         similar = data[
                #             (data['experience_level'] == experience_level) &
                #             (data['company_size'] == company_size)
                #         ]
                        
                #         if len(similar) > 0:
                #             avg_similar = similar['salary_in_usd'].mean()
                #             diff_pct = ((salary_usd - avg_similar) / avg_similar) * 100
                            
                #             st.markdown(f"""
                #             <div class="info-box">
                #                 <h4>👥 Similar Profiles</h4>
                #                 <p><strong>Found:</strong> {len(similar)} professionals</p>
                #                 <p><strong>Average Salary:</strong> ${avg_similar:,.0f}</p>
                #                 <p><strong>Your Prediction vs Average:</strong> 
                #                 <span style="color: {'green' if diff_pct > 0 else 'red'};">
                #                 {diff_pct:+.1f}%
                #                 </span>
                #                 </p>
                #             </div>
                #             """, unsafe_allow_html=True)
                    
                #     with col_y:
                #         # Experience level comparison
                #         exp_avg = data.groupby('experience_level')['salary_in_usd'].mean()
                        
                #         if experience_level in exp_avg.index:
                #             st.markdown(f"""
                #             <div class="info-box">
                #                 <h4>📊 Experience Level Averages</h4>
                #                 <p><strong>Entry (EN):</strong> ${exp_avg.get('EN', 0):,.0f}</p>
                #                 <p><strong>Mid (MI):</strong> ${exp_avg.get('MI', 0):,.0f}</p>
                #                 <p><strong>Senior (SE):</strong> ${exp_avg.get('SE', 0):,.0f}</p>
                #                 <p><strong>Executive (EX):</strong> ${exp_avg.get('EX', 0):,.0f}</p>
                #             </div>
                #             """, unsafe_allow_html=True)
                
                # # Conversion rate info
                # st.info(f"💱 **Conversion Rate Used:** 1 USD = ₹{USD_TO_INR} INR")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                with st.expander("🔧 Debug Info"):
                    st.exception(e)

# -------------------------------
# PAGE 2: MARKET ANALYSIS
# -------------------------------
elif page == "📊 Market Analysis":
    
    st.markdown("---")
    
    if data is None:
        st.error("⚠️ Dataset not loaded. Analysis unavailable.")
        st.stop()
    
    st.subheader("📊 Data Science Job Market Analysis")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Total Jobs", f"{len(data):,}")
    with col2:
        avg_usd = data['salary_in_usd'].mean()
        st.metric("💰 Avg Salary (USD)", f"${avg_usd:,.0f}")
    with col3:
        avg_inr = avg_usd * USD_TO_INR
        st.metric("💰 Avg Salary (INR)", f"₹{avg_inr:,.0f}")
    with col4:
        st.metric("🌍 Countries", data['company_location'].nunique())
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📈 By Experience", "🏢 By Company", "🌍 By Location", "📅 Trends"])
    
    with tab1:
        st.markdown("### Salary Distribution by Experience Level")
        
        exp_data = data.groupby('experience_level')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()
        exp_data['mean_inr'] = exp_data['mean'] * USD_TO_INR
        
        # Bar chart
        fig = px.bar(exp_data, x='experience_level', y='mean',
                    title='Average Salary by Experience Level (USD)',
                    labels={'mean': 'Average Salary (USD)', 'experience_level': 'Experience'},
                    color='mean', color_continuous_scale='viridis',
                    text='mean')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(
            exp_data.style.format({
                'mean': '${:,.0f}',
                'median': '${:,.0f}',
                'mean_inr': '₹{:,.0f}'
            }),
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### Company Size & Remote Work Analysis")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            size_data = data.groupby('company_size')['salary_in_usd'].mean().reset_index()
            fig = px.bar(size_data, x='company_size', y='salary_in_usd',
                        title='Average Salary by Company Size',
                        color='salary_in_usd', color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            remote_data = data.groupby('remote_ratio')['salary_in_usd'].mean().reset_index()
            fig = px.line(remote_data, x='remote_ratio', y='salary_in_usd',
                         title='Salary by Remote Work %',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Geographic Salary Analysis")
        
        country_data = data.groupby('company_location')['salary_in_usd'].agg(['mean', 'count'])
        country_data = country_data[country_data['count'] >= 10].sort_values('mean', ascending=False).head(15).reset_index()
        
        fig = px.bar(country_data, x='company_location', y='mean',
                    title='Top 15 Countries by Average Salary (min 10 jobs)',
                    labels={'mean': 'Average Salary (USD)', 'company_location': 'Country'},
                    color='mean', color_continuous_scale='reds',
                    text='mean')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Salary Trends Over Time")
        
        year_data = data.groupby('work_year')['salary_in_usd'].agg(['mean', 'count']).reset_index()
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            fig = px.line(year_data, x='work_year', y='mean',
                         title='Average Salary Trend',
                         markers=True, line_shape='spline')
            fig.update_traces(text=year_data['mean'], texttemplate='$%{text:,.0f}', textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_d:
            fig = px.bar(year_data, x='work_year', y='count',
                        title='Number of Jobs by Year',
                        color='count', color_continuous_scale='greens',
                        text='count')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PAGE 3: ABOUT
# -------------------------------
else:
    st.markdown("---")
    st.subheader("ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 About
        
        This **AI-powered salary prediction tool** helps data science professionals 
        understand their market value based on:
        
        - 📊 **Experience Level**
        - 💼 **Job Title**  
        - 🏢 **Company Size & Location**
        - 🌍 **Geographic Location**
        - 💻 **Remote Work Arrangement**
        
        ### 🤖 Technology
        
        - **Model:** Random Forest Regressor
        - **Accuracy:** ~85-90% R² Score
        - **Dataset:** 15,000+ real salary records
        - **Currency:** USD & INR supported
        
        ### 💱 Currency Conversion
        
        - Current Rate: **1 USD = ₹{} INR**
        - Rates updated periodically
        - Both currencies displayed for convenience
        """.format(USD_TO_INR))
    
    with col2:
        st.markdown("""
        ### 📈 Features
        
        ✅ **Real-time Predictions**  
        ✅ **Dual Currency Support** (USD & INR)  
        ✅ **Market Analysis Dashboard**  
        ✅ **Salary Breakdowns** (Annual, Monthly, Hourly)  
        ✅ **Percentile Rankings**  
        ✅ **Interactive Visualizations**  
        ✅ **Similar Profile Comparisons**  
        
        ### 🎓 Use Cases
        
        - **Job Seekers:** Know your market value
        - **Employers:** Set competitive salaries
        - **Career Planning:** Understand growth potential
        - **Salary Negotiation:** Data-backed expectations
        
        ### ⚠️ Disclaimer
        
        Predictions are based on historical data and market trends. 
        Actual salaries may vary based on individual skills, 
        negotiation, and company-specific factors.
        
        ---
        
        **Created for DS Lab Course Project 2024**  
        **Powered by Random Forest ML Model**
        """)
    
    st.markdown("---")
    st.info("💡 **Tip:** For best results, ensure your inputs match real job market terminology and use standard country codes.")