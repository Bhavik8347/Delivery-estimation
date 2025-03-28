import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("linear_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Estimated Delivery Time Predictor")

# User Inputs
customer_zip = st.number_input("Enter Customer Zip Code:", min_value=1000, max_value=99999, step=1)
product_weight = st.number_input("Enter Product Weight (g):", min_value=1, max_value=50000, step=1)
product_length = st.number_input("Enter Product Length (cm):", min_value=1, max_value=200, step=1)
product_height = st.number_input("Enter Product Height (cm):", min_value=1, max_value=200, step=1)
product_width = st.number_input("Enter Product Width (cm):", min_value=1, max_value=200, step=1)
seller_zip = st.number_input("Enter Seller Zip Code:", min_value=1000, max_value=99999, step=1)

if st.button("Predict Delivery Time"):
    # Prepare input for model
    input_data = np.array([[customer_zip, product_weight, product_length, product_height, product_width, seller_zip]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict
    predicted_days = model.predict(input_data_scaled)[0]
    st.success(f"Estimated Delivery Time: {predicted_days:.2f} days")