import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Predictor")
st.markdown("Fill in the car details below to get an estimated price.")
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1E3A5F, #2E86C1);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: white;
    }
    .metric-box h1 { font-size: 2.8rem; margin: 0; }
    .metric-box p  { font-size: 1rem;  margin: 4px 0 0; opacity: 0.85; }
    .section-label { font-size: 0.78rem; font-weight: 600;
                     text-transform: uppercase; color: #888; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_options():
    df = pd.read_csv("artifacts/data.csv")
    type_brand_map = (
        df.groupby("Model_Brand")["Model_Type"]
        .unique()
        .apply(sorted)
        .to_dict()
    )
    return type_brand_map
type_brand_map = load_options()
with st.form('car_price_form'):
    st.markdown('<p class="section-label">🔑 Basic Information</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Model_Brand = c1.selectbox("Model Brand", options=sorted(type_brand_map.keys()))
    Model_Type = c2.selectbox("Model Type", options=type_brand_map[Model_Brand])
    Car_Age      = c3.number_input("Car Age (years)", min_value=0, max_value=25, value=5)
    Fuel_Type    = c1.selectbox("Fuel Type", ["Petrol", "Electric", "Hybrid"])
    Transmission = c2.selectbox("Transmission", ["Automatic", "Manual"])
    Condition    = c3.selectbox("Condition", ["Good", "Fair", "Excellent"])
    Color         =c1.selectbox("Color", ["Brown","Black", "White", "Silver", "Beige", "Yellow"])

    st.divider()

    st.markdown('<p class="section-label">🔑Technical Specifications</p>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    Engine_Size      = t1.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0)
    DoorsNum         = t2.number_input("Number of Doors", min_value=2, max_value=6, value=4)
    Weight           = t3.number_input("Weight (kg)", min_value=500, max_value=5000, value=1500)
    carlength        = t4.number_input("Car Length (mm)", min_value=2000, max_value=6000, value=4500)
    carwidth         = t1.number_input("Car Width (mm)", min_value=1000, max_value=3000, value=1800)
    peakrpm          = t2.number_input("Peak RPM", min_value=3000, max_value=10000, value=5500)
    Cylinder_Numbers = t3.selectbox("Cylinders", ['five', 'four', 'six', 'three', 'two'])
    Wheel        = t4.selectbox("Wheels", ["Chrome", "Steel", "Alloy",'Forged','Hubcaps'])

    st.divider()
    st.markdown('<p class="section-label">🔑 Usage & History</p>', unsafe_allow_html=True)
    u1,u2,u3 = st.columns(3)
    Owners            = u1.number_input("Previous Owners", min_value=0, max_value=10, value=1)
    Estimated_Mileage = u2.number_input("Estimated Mileage (km)", min_value=0, value=50000)
    monthly_mileage   = u3.number_input("Monthly Mileage (km)", min_value=0, value=1500)
    Warranty          = u1.number_input("Warranty (years)", min_value=0, max_value=10, value=1)
    TAge              = u2.selectbox("Tyre Age", [1, 2, 3, 4, 5]) 
    Credit_History    = u3.selectbox("Credit History", ["Good", "Average", "Poor"]) 
    Insurance         = u1.selectbox("Insurance", ["No insurance", "Comprehensive", "Third Party","Collision"])
    Service_History   = u2.selectbox("Service History", ["Full Service", "Partial Service", "No Service"])
    Safety       = u3.selectbox("Safety Ratings", ["5 stars", "4 stars", "3 stars", "2 stars", "1 star",'Not rated/Unknown'])

    st.divider()
    st.markdown('<p class="section-label">🔑 Additional Features</p>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    Cruise       = f1.radio("Cruise Control", ["Yes", "No"],horizontal=True)
    Leather_Seats = f2.radio("Leather Seats", ["Yes", "No"],horizontal=True)
    Heated_Seats = f3.radio("Heated Seats", ["Yes", "No"],horizontal=True)
    Navigation   = f4.radio("Navigation", ["Yes", "No"],horizontal=True)
    Premium_Sound = f1.radio("Premium Sound", ["Yes", "No"],horizontal=True)
    Multimedia   = f2.radio("Multimedia", ["Yes", "No"],horizontal=True)
    Bluetooth    = f3.radio("Bluetooth", ["Yes", "No"],horizontal=True)
    Sunroof      = f4.radio("Sunroof", ["Yes", "No"],horizontal=True)

    st.divider()
    submitted = st.form_submit_button("🔍 Predict Car Price", use_container_width=True)
    #PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    try:
        data = CustomData(
            DoorsNum=DoorsNum, Owners=Owners, Warranty=Warranty,
            Engine_Size=Engine_Size, Weight=Weight, carlength=carlength,
            carwidth=carwidth, monthly_mileage=monthly_mileage, peakrpm=peakrpm,
            Estimated_Mileage=Estimated_Mileage, Car_Age=Car_Age,
            Model_Type=Model_Type, Model_Brand=Model_Brand, Fuel_Type=Fuel_Type, 
            Transmission=Transmission, Condition=Condition, Color=Color, Cruise=Cruise,
            Leather_Seats=Leather_Seats, Heated_Seats=Heated_Seats,
            Navigation=Navigation, Insurance=Insurance,
            Service_History=Service_History, Safety=Safety,
            Premium_Sound=Premium_Sound, Multimedia=Multimedia,
            Bluetooth=Bluetooth, Wheel=Wheel, Sunroof=Sunroof,
            TAge=TAge, Cylinder_Numbers=Cylinder_Numbers,
            Credit_History=Credit_History,
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result = pipeline.predict(df)

        st.success(f"💰 Estimated Car Price: ${result[0]:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")