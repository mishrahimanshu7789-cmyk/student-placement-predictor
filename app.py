
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the trained model and feature names
if not os.path.exists("placement_model.pkl") or not os.path.exists("feature_columns.pkl"):
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

model = joblib.load("placement_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Application title
st.title("🎓 Student Placement Predictor")
st.write("Predict student placement using Logistic Regression.")
st.info("Enter the student's academic details and skills below.")

# Input form
with st.form("placement_form"):

    st.subheader("Academic Details")

    cgpa = st.slider(
        "CGPA", min_value=4.0, max_value=10.0,
        value=7.0, step=0.1
    )

    tenth = st.slider(
        "10th Percentage", min_value=0.0, max_value=100.0,
        value=75.0, step=0.1
    )

    twelfth = st.slider(
        "12th Percentage", min_value=0.0, max_value=100.0,
        value=75.0, step=0.1
    )

    st.subheader("Skills and Experience")

    internships = st.number_input(
        "Number of Internships",
        min_value=0, max_value=5, value=1, step=1
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0, max_value=9, value=2, step=1
    )

    coding = st.slider(
        "Coding Skills (0-10)",
        min_value=0.0, max_value=10.0,
        value=6.0, step=0.1
    )

    communication = st.slider(
        "Communication Skills (2-10)",
        min_value=2.0, max_value=10.0,
        value=6.5, step=0.1
    )

    submitted = st.form_submit_button(
        "Predict Placement",
        use_container_width=True
    )

# Make prediction
if submitted:

    # Create input data using the original column names
    input_data = pd.DataFrame(
        [[cgpa, tenth, twelfth, internships,
          projects, coding, communication]],
        columns=[
            "CGPA",
            "10th_Percentage",
            "12th_Percentage",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Communication_Skills"
        ]
    )

    # Ensure the input order matches training
    input_data = input_data[feature_columns]

    # Predict class and probability
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    placement_probability = probabilities[
        list(model.classes_).index(1)
    ]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated Placement Probability",
        f"{placement_probability * 100:.2f}%"
    )

    st.progress(float(placement_probability))

    if prediction == 1:
        st.success("The model predicts: LIKELY TO BE PLACED")
    else:
        st.warning("The model predicts: NOT LIKELY TO BE PLACED")

    st.caption(
        "This is a machine-learning estimate based on the supplied "
        "dataset, not a guarantee of actual placement."
    )
