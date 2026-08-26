import streamlit as st
import pandas as pd
import joblib

diabetes_model = joblib.load(
    "models/diabetes_model.pkl"
)

heart_model = joblib.load(
    "models/heart_model.pkl"
)

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI-Based Disease Prediction System")

st.markdown(
    """
    ### Machine Learning Based Healthcare Prediction

    This application uses trained machine learning models
    to estimate the likelihood of **Diabetes** and
    **Heart Disease** based on the entered health parameters.
    """
)

st.warning(
    "⚠️ This application is developed for academic and "
    "educational purposes only. It is not a medical diagnosis."
)


st.sidebar.title("Navigation")

disease = st.sidebar.radio(
    "Select Prediction",
    [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction"
    ]
)

if disease == "Home":

    st.header("Welcome!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🩺 Diabetes Prediction")
        st.write(
            "Predict the likelihood of diabetes using "
            "patient health parameters."
        )

    with col2:
        st.subheader("❤️ Heart Disease Prediction")
        st.write(
            "Predict the likelihood of heart disease using "
            "patient health parameters."
        )

    st.divider()

    st.info(
        "Select a prediction module from the sidebar "
        "to begin."
    )

elif disease == "Diabetes Prediction":

    st.header("🩺 Diabetes Prediction")

    st.write(
        "Enter the patient's health information below."
    )

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.divider()

    if st.button(
        "🔍 Predict Diabetes",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        prediction = diabetes_model.predict(
            input_data
        )[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Higher likelihood of Diabetes"
            )

        else:

            st.success(
                "✅ Lower likelihood of Diabetes"
            )

        # Probability
        if hasattr(
            diabetes_model,
            "predict_proba"
        ):

            probability = (
                diabetes_model
                .predict_proba(input_data)[0][1]
            )

            st.metric(
                "Model Estimated Probability",
                f"{probability * 100:.2f}%"
            )


elif disease == "Heart Disease Prediction":

    st.header("❤️ Heart Disease Prediction")

    st.write(
        "Enter the patient's cardiovascular health "
        "parameters below."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50.0,
            max_value=250.0,
            value=120.0
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50.0,
            max_value=700.0,
            value=200.0
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            ["No", "Yes"]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50.0,
            max_value=250.0,
            value=150.0
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels (ca)",
            [0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    st.divider()

    if st.button(
        "🔍 Predict Heart Disease",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [[
                age,
                1 if sex == "Male" else 0,
                cp,
                trestbps,
                chol,
                1 if fbs == "Yes" else 0,
                restecg,
                thalach,
                1 if exang == "Yes" else 0,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        )

        prediction = heart_model.predict(
            input_data
        )[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Higher likelihood of Heart Disease"
            )

        else:

            st.success(
                "✅ Lower likelihood of Heart Disease"
            )

        # Probability
        if hasattr(
            heart_model,
            "predict_proba"
        ):

            probability = (
                heart_model
                .predict_proba(input_data)[0][1]
            )

            st.metric(
                "Model Estimated Probability",
                f"{probability * 100:.2f}%"
            )


st.divider()

st.caption(
    "Disease Prediction System | Machine Learning Major Project"
)
