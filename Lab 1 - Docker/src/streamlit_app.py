import streamlit as st
import requests

st.title("🐧 Penguin Species Classifier")

st.write("Enter penguin measurements to predict the species")

# Input fields
island = st.selectbox("Island", options=[0, 1, 2], format_func=lambda x: ["Biscoe", "Dream", "Torgersen"][x])
culmen_length = st.slider("Culmen Length (mm)", 30.0, 60.0, 39.1)
culmen_depth = st.slider("Culmen Depth (mm)", 13.0, 22.0, 18.7)
flipper_length = st.slider("Flipper Length (mm)", 170.0, 230.0, 181.0)
body_mass = st.slider("Body Mass (g)", 2500.0, 6500.0, 3750.0)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: ["Female", "Male"][x])

if st.button("Predict Species"):
    # Prepare data
    data = {
        "island": island,
        "culmen_length_mm": culmen_length,
        "culmen_depth_mm": culmen_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
        "sex": sex
    }
    
    # Send to Flask API
    try:
        response = requests.post("http://flask-api:8000/predict", json=data)
        result = response.json()
        
        if 'species' in result:
            st.success(f"Predicted Species: **{result['species']}**")
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Could not connect to API: {str(e)}")