import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir,
        "diabetes_gradient_boosting_model.pkl"
    )

    preprocessor_path = os.path.join(
        base_dir,
        "diabetes_preprocessor.pkl"
    )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


model, preprocessor = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    /* Header */
    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 50%,
            #374151 100%
        );
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #d1d5db;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        margin-bottom: 15px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }

    /* Result */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #e5e7eb;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
    }

    .probability {
        font-size: 48px;
        font-weight: 900;
        margin: 10px 0;
    }

    .small-text {
        color: #6b7280;
        font-size: 13px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        height: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.caption("Machine Learning Prediction System")

    st.divider()

    st.markdown("### System")

    st.success("Model Loaded")

    st.markdown(
        """
        **Algorithm**

        HistGradientBoostingClassifier

        **Task**

        Binary Classification

        **Output**

        Diabetes Probability
        """
    )

    st.divider()

    st.markdown("### Model Performance")

    st.metric("Accuracy", "97.16%")
    st.metric("ROC-AUC", "0.9771")
    st.metric("Precision", "98.08%")

    st.divider()

    st.caption(
        "For educational and research purposes only. "
        "This system is not a medical diagnosis."
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Diabetes AI Predictor
        </div>

        <div class="hero-subtitle">
            Machine Learning powered diabetes risk prediction
            using demographic, lifestyle and clinical indicators.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="card">
            <div class="metric-title">MODEL</div>
            <div class="metric-value">HGB</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="card">
            <div class="metric-title">ACCURACY</div>
            <div class="metric-value">97.16%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value">0.9771</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        """
        <div class="card">
            <div class="metric-title">INPUT FEATURES</div>
            <div class="metric-value">8</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PATIENT INPUT
# =========================================================

st.subheader("Patient Assessment")

st.write(
    "Enter the patient's demographic, lifestyle and clinical information."
)

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # DEMOGRAPHIC INFORMATION
    # -----------------------------------------------------

    with col1:

        st.markdown("#### Demographics")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=1.0,
            max_value=80.0,
            value=30.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

    # -----------------------------------------------------
    # MEDICAL / LIFESTYLE INFORMATION
    # -----------------------------------------------------

    with col2:

        st.markdown("#### Clinical & Lifestyle")

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "Never Smoked",
                "Former Smoker",
                "Ever Smoked",
                "Not Currently Smoking",
                "Current Smoker",
                "Smoking History Unknown"
            ]
        )

    st.divider()

    # -----------------------------------------------------
    # CLINICAL MEASUREMENTS
    # -----------------------------------------------------

    st.markdown("#### Clinical Measurements")

    c1, c2 = st.columns(2)

    with c1:

        hba1c = st.number_input(
            "HbA1c Level (%)",
            min_value=3.5,
            max_value=9.0,
            value=5.5,
            step=0.1
        )

    with c2:

        glucose = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=80,
            max_value=300,
            value=140,
            step=1
        )

    st.divider()

    predict = st.form_submit_button(
        "🔍  ANALYZE DIABETES RISK",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    patient_data = pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        }
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if age <= 0:
        st.error("Age must be greater than zero.")

    elif bmi <= 0:
        st.error("BMI must be greater than zero.")

    elif hba1c <= 0:
        st.error("HbA1c value must be valid.")

    elif glucose <= 0:
        st.error("Blood glucose value must be valid.")

    else:

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        processed_data = preprocessor.transform(patient_data)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(processed_data)[0]

        probability = model.predict_proba(processed_data)[0][1]

        probability_percent = probability * 100

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-title">
                        Diabetes Risk Detected
                    </div>

                    <div class="probability">
                        {probability_percent:.2f}%
                    </div>

                    <div>
                        Predicted probability of diabetes
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-title">
                        Lower Diabetes Risk
                    </div>

                    <div class="probability">
                        {probability_percent:.2f}%
                    </div>

                    <div>
                        Predicted probability of diabetes
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.markdown("#### Diabetes Probability")

        st.progress(
            min(probability, 1.0)
        )

        st.caption(
            f"Model probability: {probability_percent:.2f}%"
        )

        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.divider()

        st.subheader("Patient Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Age", f"{age:.0f}")

        with s2:
            st.metric("BMI", f"{bmi:.1f}")

        with s3:
            st.metric("HbA1c", f"{hba1c:.1f}%")

        with s4:
            st.metric("Glucose", f"{glucose} mg/dL")

        # -------------------------------------------------
        # INPUT DETAILS
        # -------------------------------------------------

        with st.expander("View Patient Input Data"):

            display_data = patient_data.copy()

            display_data["hypertension"] = display_data[
                "hypertension"
            ].map(
                {0: "No", 1: "Yes"}
            )

            display_data["heart_disease"] = display_data[
                "heart_disease"
            ].map(
                {0: "No", 1: "Yes"}
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.divider()

        st.subheader("Model Interpretation")

        if probability < 0.20:

            st.info(
                "The model estimates a relatively low probability "
                "of diabetes for the provided inputs."
            )

        elif probability < 0.50:

            st.warning(
                "The model estimates a moderate probability "
                "of diabetes for the provided inputs."
            )

        elif probability < 0.75:

            st.warning(
                "The model estimates an elevated probability "
                "of diabetes for the provided inputs."
            )

        else:

            st.error(
                "The model estimates a high probability "
                "of diabetes for the provided inputs."
            )

        st.caption(
            "This prediction is generated by a machine learning model "
            "and should not be interpreted as a clinical diagnosis."
        )


# =========================================================
# ABOUT MODEL
# =========================================================

st.divider()

with st.expander("About This Machine Learning Model"):

    st.markdown(
        """
        ### Diabetes Prediction Model

        This application uses a supervised machine learning
        classification model trained to estimate the probability
        of diabetes.

        **Model:** HistGradientBoostingClassifier

        **Input features:**

        - Gender
        - Age
        - Hypertension
        - Heart Disease
        - Smoking History
        - BMI
        - HbA1c Level
        - Blood Glucose Level

        **Model evaluation:**

        - Accuracy: 97.16%
        - Precision: 98.08%
        - F1 Score: 0.8112
        - ROC-AUC: 0.9771

        The preprocessing pipeline uses numerical scaling and
        categorical one-hot encoding before prediction.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="text-align:center; padding:30px; color:#6b7280;">

        <b>Diabetes AI Predictor</b><br>
        Machine Learning • Python • Scikit-learn • Streamlit

        <br><br>

        Educational / Research Application

    </div>
    """,
    unsafe_allow_html=True
)