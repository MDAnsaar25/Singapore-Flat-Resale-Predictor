import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")

# Inject background + button style
st.markdown("""
    <style>
    .stApp {
        background-color: #d6dbdf;
    }
    .main > div {
        background-color: #ffffff;
        padding: 2rem 2rem;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.05);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and features
model = joblib.load("xgb_resale_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>🏠 Flat Resale Price Estimator</h1>", unsafe_allow_html=True)
st.markdown("🔍 Use this tool to estimate the resale value of a Singapore HDB flat based on its details.")

# Sidebar info
st.sidebar.title("📘 About")
st.sidebar.markdown("""
This app uses a machine learning model (XGBoost Regressor) trained on historical HDB resale data.
""")
st.sidebar.markdown("**Features considered:**")
st.sidebar.markdown("""
- 📍 Town  
- 🏢 Flat Type & Model  
- 📐 Floor Area  
- 📅 Lease Commencement  
- 🔢 Storey Range  
""")

# Input form
st.subheader("🔧 Input Flat Details")
col1, col2 = st.columns(2)

with col1:
    town = st.selectbox("📍 Select Town", sorted([
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", "BUKIT TIMAH",
        "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG",
        "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN"
    ]))
    flat_type = st.selectbox("🏢 Flat Type", ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI GENERATION"])
    flat_model = st.selectbox("🏷️ Flat Model", [
        "IMPROVED", "NEW GENERATION", "MODEL A", "STANDARD", "SIMPLIFIED", "MAISONETTE",
        "APARTMENT", "PREMIUM APARTMENT", "TYPE S1", "TYPE S2"
    ])

with col2:
    floor_area = st.slider("📐 Floor Area (sqm)", min_value=30, max_value=200, value=90)
    storey_range = st.selectbox("📊 Storey Range", [
        "01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18", "19 TO 21",
        "22 TO 24", "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36", "37 TO 39", "40 TO 42",
        "43 TO 45", "46 TO 48", "49 TO 51"
    ])
    lease_commence = st.slider("📅 Lease Commencement Year", min_value=1960, max_value=2025, value=2005)

# Feature engineering
def get_storey_median(range_str):
    low, high = range_str.split(' TO ')
    return (int(low) + int(high)) / 2

flat_age = 2025 - lease_commence
storey_median = get_storey_median(storey_range)

input_data = {
    "floor_area_sqm": floor_area,
    "flat_age": flat_age,
    "storey_median": storey_median
}

for col in feature_columns:
    if col.startswith("town_"):
        input_data[col] = 1 if col == f"town_{town}" else 0
    elif col.startswith("flat_type_"):
        input_data[col] = 1 if col == f"flat_type_{flat_type}" else 0
    elif col.startswith("flat_model_"):
        input_data[col] = 1 if col == f"flat_model_{flat_model}" else 0
    elif col not in input_data:
        input_data[col] = 0

# Prediction section
st.markdown("---")
if st.button("📈 Predict Resale Price"):
    input_df = pd.DataFrame([input_data])[feature_columns]
    predicted_price = model.predict(input_df)[0]

    st.success(f"💰 **Estimated Resale Price: SGD {predicted_price:,.2f}**")

    # 🏠 Animated house balloons - fly to top of screen
    st.markdown("""
    <div style="position: relative; height: 180px;">
        <div class="house-fly" style="left: 5%;  animation-delay: 0s;">🏠</div>
        <div class="house-fly" style="left: 15%; animation-delay: 0.2s;">🏠</div>
        <div class="house-fly" style="left: 25%; animation-delay: 0.4s;">🏠</div>
        <div class="house-fly" style="left: 35%; animation-delay: 0.6s;">🏠</div>
        <div class="house-fly" style="left: 45%; animation-delay: 0.8s;">🏠</div>
        <div class="house-fly" style="left: 55%; animation-delay: 1s;">🏠</div>
        <div class="house-fly" style="left: 65%; animation-delay: 1.2s;">🏠</div>
        <div class="house-fly" style="left: 75%; animation-delay: 1.4s;">🏠</div>
        <div class="house-fly" style="left: 85%; animation-delay: 1.6s;">🏠</div>
        <div class="house-fly" style="left: 95%; animation-delay: 1.8s;">🏠</div>
    </div>

    <style>
    .house-fly {
        position: absolute;
        bottom: 0;
        font-size: 36px;
        opacity: 1;
        animation: flyUp 3s ease-out forwards;
    }

    @keyframes flyUp {
        0%   { bottom: 0px; opacity: 1; }
        80%  { opacity: 1; }
        100% { bottom: 100vh; opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("_🚀 Powered by XGBoost & Streamlit_")
