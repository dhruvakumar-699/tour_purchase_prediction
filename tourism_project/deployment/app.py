import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Dhruva23101995/tour_purchase_prediction", filename="best_tour_purchase_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("tour Purchase Prediction App")
st.write("""
This application predicts the likelihood of a tour purchase based on its parameters.
Please enter the customer details below to get a prediction.
""")

# User input
Type = st.selectbox("Customer Type", ["Standard", "Deluxe", "Super Deluxe"])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=60, value=15)
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.number_input("Number Of Trips per Year", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Has Passport?", [0, 1])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=500000, value=30000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'Type': Type
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "The customer is likely to PURCHASE the package" if prediction == 1 else "The customer is NOT likely to purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
