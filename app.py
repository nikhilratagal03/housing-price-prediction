import pickle
import numpy as np
import streamlit as st

# -------------------
# Load trained model
# -------------------
with open("xgboost2_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

st.title("🏡 King County House Price Prediction")
st.markdown("Fill in the details below to estimate the selling price of a house.")

# -------------------
# Define feature inputs (match your training data order)
# -------------------
# Categorical feature choices based on dataset
waterfront_options = [0, 1]  # 0 = No, 1 = Yes
view_options = [0, 1, 2, 3, 4]
condition_options = [1, 2, 3, 4, 5]
grade_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
zipcode_options = [
    98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011,
    98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031,
    98032, 98033, 98034, 98038, 98039, 98040, 98042, 98045, 98052, 98053,
    98055, 98056, 98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077,
    98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115,
    98116, 98117, 98118, 98119, 98122, 98125, 98126, 98133, 98136, 98144,
    98146, 98148, 98155, 98166, 98168, 98177, 98178, 98188, 98198, 98199
]

st.subheader("Basic House Details")
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=2000)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=0, value=5000)
floors = st.number_input("Floors", min_value=0.0, max_value=4.0, value=1.0, step=0.5)
waterfront = st.selectbox("Waterfront", waterfront_options)
view = st.selectbox("View Quality (0-4)", view_options)
condition = st.selectbox("Condition (1-5)", condition_options)
grade = st.selectbox("Grade (1-13)", grade_options)

st.subheader("Construction Details")
sqft_above = st.number_input("Sqft Above Ground", min_value=0, value=1500)
sqft_basement = st.number_input("Sqft Basement", min_value=0, value=500)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)

st.subheader("Location Details")
zipcode = st.selectbox("Zipcode", zipcode_options)
lat = st.number_input("Latitude", value=47.5112)
long = st.number_input("Longitude", value=-122.257)
sqft_living15 = st.number_input("Living Area (15 Nearest Houses)", min_value=0, value=2000)
sqft_lot15 = st.number_input("Lot Size (15 Nearest Houses)", min_value=0, value=5000)

# -------------------
# Predict Button
# -------------------
if st.button("Predict Price 💰"):
    features = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, grade, sqft_above,
        sqft_basement, yr_built, yr_renovated, zipcode, lat,
        long, sqft_living15, sqft_lot15
    ]
    X = np.array(features).reshape(1, -1)
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # Reverse log transform

    st.markdown(f"### 🏠 Estimated Price: **${y_pred[0]:,.2f}**")
    st.balloons()
