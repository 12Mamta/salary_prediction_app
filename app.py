import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('model.pkl')  # Ensure this file is in the same folder

# Set the app title
st.title("💼 Employee Salary Prediction")
st.markdown("### Please enter the following information:")

# User inputs
age = st.number_input("Enter Age", min_value=17, max_value=90, value=30)
education_num = st.slider("Education Number", min_value=1, max_value=16, value=10)
hours_per_week = st.slider("Hours Worked per Week", min_value=1, max_value=100, value=40)

# When user clicks the button
if st.button("🔍 Predict Salary"):
    # Default/fixed values for other features
    default_features = [
        1,     # workclass_encoded (e.g., Private)
        4,     # marital_status_encoded (e.g., Married)
        0,     # occupation_encoded (e.g., Tech-support)
        1,     # relationship_encoded (e.g., Husband)
        0,     # race_encoded (e.g., White)
        1,     # sex_encoded (1 = Male)
        0,     # native_country_encoded (e.g., United States)
        0.0,   # capital_gain
        0.0,   # capital_loss
        40.0,  # avg hours per week (repetition of user input allowed)
        0      # dummy/optional feature (you can remove this if not needed)
    ]

    # Final input vector
    input_data = np.array([[age, education_num, hours_per_week] + default_features])

    # Make prediction
    prediction = model.predict(input_data)

    # Show prediction
    if prediction[0] == 1:
        st.success("🎯 Prediction: Salary > 50K 💰")
    else:
        st.info("📉 Prediction: Salary ≤ 50K")
