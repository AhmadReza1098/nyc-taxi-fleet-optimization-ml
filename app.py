import streamlit as st
import pandas as pd
import joblib

# Set up the page layout
st.set_page_config(page_title="NYC Taxi Fleet AI", page_icon="🚕", layout="wide")

# --- CACHE THE MODELS SO THE APP IS FAST ---
@st.cache_resource 
def load_all_models():
    # Load Tip Model
    tip_model = joblib.load('taxi_tip_model.pkl')
    tip_cols = joblib.load('model_columns.pkl')
    
    # Load Duration Model
    duration_model = joblib.load('duration_model.pkl')
    duration_cols = joblib.load('duration_columns.pkl')
    
    # Load Destination Model
    dest_model = joblib.load('destination_model.pkl')
    dest_cols = joblib.load('destination_columns.pkl')
    
    return tip_model, tip_cols, duration_model, duration_cols, dest_model, dest_cols

tip_model, tip_cols, dur_model, dur_cols, dest_model, dest_cols = load_all_models()

# Global UI Dictionaries
boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown"]
day_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}

# --- APP HEADER ---
st.title("🚕 NYC Taxi Fleet Optimization Suite")
st.markdown("An end-to-end Machine Learning platform for predictive fleet routing, duration estimation, and revenue forecasting.")
st.divider()

# --- CREATE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "💸 Tip Predictor", "⏱️ Duration Estimator", "🔮 Destination AI"])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.header("Project Overview")
    st.markdown("""
    Welcome to the Fleet Optimization Dashboard. This application contains three distinct machine learning models built on millions of rows of NYC Taxi data, engineered via DuckDB:
    
    * **💸 Tip Predictor (XGBoost):** Forecasts expected gratuity to identify high-yield driver routes, combating distance-based heteroskedasticity.
    * **⏱️ Duration Estimator (XGBoost):** Maps the non-linear physics of NYC traffic to predict exact trip durations within a 4-minute margin of error.
    * **🔮 Destination AI (Random Forest):** Anticipates human commuting behaviors to predict final drop-off boroughs, utilizing SMOTE to counteract massive intra-Manhattan class imbalance.
    """)

# ==========================================
# TAB 2: TIP PREDICTOR
# ==========================================
with tab2:
    st.header("💸 Revenue Predictor")
    col1, col2 = st.columns(2)
    with col1:
        t_distance = st.number_input("Trip Distance (Miles)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, key="t_dist")
        t_hour = st.slider("Pickup Hour", 0, 23, 14, key="t_hour")
        t_day = st.selectbox("Day of the Week", list(day_map.keys()), key="t_day")
    with col2:
        t_pickup = st.selectbox("Pickup Borough", boroughs, index=0, key="t_pick")
        t_dropoff = st.selectbox("Dropoff Borough", boroughs, index=0, key="t_drop")

    if st.button("🔮 Predict Expected Tip", type="primary"):
        input_dict = {col: 0 for col in tip_cols}
        input_dict['trip_distance'] = t_distance
        input_dict['pickup_hour'] = t_hour
        input_dict['day_of_week'] = day_map[t_day]
        if f'pickup_borough_{t_pickup}' in input_dict: input_dict[f'pickup_borough_{t_pickup}'] = 1
        if f'dropoff_borough_{t_dropoff}' in input_dict: input_dict[f'dropoff_borough_{t_dropoff}'] = 1
        
        pred = tip_model.predict(pd.DataFrame([input_dict]))[0]
        st.success(f"### 🎯 Expected Tip: ${pred:.2f}")

# ==========================================
# TAB 3: DURATION ESTIMATOR
# ==========================================
with tab3:
    st.header("⏱️ Traffic & Duration AI")
    col3, col4 = st.columns(2)
    with col3:
        d_distance = st.number_input("Trip Distance (Miles)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, key="d_dist")
        d_hour = st.slider("Pickup Hour", 0, 23, 17, key="d_hour") # Default to rush hour!
        d_day = st.selectbox("Day of the Week", list(day_map.keys()), key="d_day")
    with col4:
        d_pickup = st.selectbox("Pickup Borough", boroughs, index=1, key="d_pick")
        d_dropoff = st.selectbox("Dropoff Borough", boroughs, index=0, key="d_drop")

    if st.button("⏱️ Estimate Trip Time", type="primary"):
        input_dict = {col: 0 for col in dur_cols}
        input_dict['trip_distance'] = d_distance
        input_dict['pickup_hour'] = d_hour
        input_dict['day_of_week'] = day_map[d_day]
        if f'pickup_borough_{d_pickup}' in input_dict: input_dict[f'pickup_borough_{d_pickup}'] = 1
        if f'dropoff_borough_{d_dropoff}' in input_dict: input_dict[f'dropoff_borough_{d_dropoff}'] = 1
        
        pred = dur_model.predict(pd.DataFrame([input_dict]))[0]
        st.info(f"### 🚦 Estimated Duration: {pred:.1f} Minutes")

# ==========================================
# TAB 4: DESTINATION PREDICTOR
# ==========================================
with tab4:
    st.header("🔮 Behavioral Destination Predictor")
    st.markdown("*Note: This model uses pre-trip logistics to forecast fleet movement and mitigate deadhead risk.*")
    
    col5, col6 = st.columns(2)
    with col5:
        dest_hour = st.slider("Pickup Hour", 0, 23, 8, key="dest_hour") # Default to morning commute
        dest_day = st.selectbox("Day of the Week", list(day_map.keys()), key="dest_day")
    with col6:
        dest_pickup = st.selectbox("Pickup Borough", boroughs, index=1, key="dest_pick") # Default Brooklyn

    if st.button("🗺️ Predict Destination", type="primary"):
        input_dict = {col: 0 for col in dest_cols}
        input_dict['pickup_hour'] = dest_hour
        input_dict['day_of_week'] = day_map[dest_day]
        if f'pickup_borough_{dest_pickup}' in input_dict: input_dict[f'pickup_borough_{dest_pickup}'] = 1
        
        pred = dest_model.predict(pd.DataFrame([input_dict]))[0]
        st.warning(f"### 📍 Predicted Dropoff Zone: {pred}")