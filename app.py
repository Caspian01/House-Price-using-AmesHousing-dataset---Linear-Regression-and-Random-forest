import joblib
import streamlit as st
import numpy as np

st.header("House Pricing using Machine Learning")

models = {
    "Linear" : 'Linear_house_price_model.pkl',
    "Random Forest" : 'Random_forest_house_price_model.pkl'
}

model_choice = st.selectbox("Pick a model", models.keys())


overall_qual = st.text_input("Overall Quality (1-10):")
gr_liv_area = st.text_input("Above ground living area in square feet: ")
garage_cars = st.text_input("Number of cars that fit in the garage: ")
garage_area = st.text_input("Garage area in square feet: ")
total_bsmt_sf = st.text_input("Total basement area in square feet: ")
full_bath = st.text_input("Number of full bathrooms: ")
year_built = st.text_input("Year the house was built: ")

if st.button("Price me!"):
    try:
        overall_qual = float(overall_qual)
        gr_liv_area = float(gr_liv_area)
        garage_cars = float(garage_cars)
        garage_area =  float(garage_area)
        total_bsmt_sf = float(total_bsmt_sf)
        full_bath =  float(full_bath)
        year_built = float(year_built)

        #Loading model
        model_selected = joblib.load(models[model_choice])
        print([overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf, full_bath, year_built])
        features = np.array([[overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf, full_bath, year_built]])
        predicted_price = model_selected.predict(features)

        st.write(f"The price of the house using {model_choice} is: {predicted_price[0]:,.2f}")

    except ValueError:
        st.error("Please insert numerical values to proceed")