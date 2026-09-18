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
        "diabetes_hist_gradient_boosting_model.pkl"
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
# RISK SPECTRUM HELPERS
# =========================================================

def risk_band(probability):
    """Return (label, color, message) for a probability in [0, 1]."""

    if probability < 0.20:
        return (
            "Low",
            "#2DE1C2",
            "The model places this patient near the calm end of the "
            "spectrum. Routine screening intervals are enough."
        )

    if probability < 0.50:
        return (
            "Moderate",
            "#7C5CFF",
            "The model sees mixed signals. Worth re-checking HbA1c and "
            "fasting glucose at the next visit."
        )

    if probability < 0.75:
        return (
            "Elevated",
            "#FFB020",
            "Several inputs are pushing the estimate upward. A confirmatory "
            "lab panel is the sensible next step."
        )

    return (
        "High",
        "#FF5D73",
        "The model estimates a high probability of diabetes. Send these "
        "numbers to a clinician for proper testing."
    )


# =========================================================
# DESIGN SYSTEM
# =========================================================
#
#   Base      #120E32  deep indigo, the console background
#   Panel     #1C1650  raised surface
#   Mint      #2DE1C2  low risk
#   Violet    #7C5CFF  moderate risk / brand
#   Amber     #FFB020  elevated risk
#   Coral     #FF5D73  high risk
#
#   Display   Bricolage Grotesque
#   Body/UI   Inter Tight
#
#   The four accents are not decoration: they are a single ordered
#   spectrum that runs from low to high probability, and every colored
#   element on the page reads off that same scale.
# =========================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Inter+Tight:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>

    :root {
        --base:    #120E32;
        --panel:   #1C1650;
        --panel-2: #241C66;
        --line:    rgba(160, 145, 255, 0.18);
        --ink:     #F2EFFF;
        --muted:   #A79FDD;
        --mint:    #2DE1C2;
        --violet:  #7C5CFF;
        --amber:   #FFB020;
        --coral:   #FF5D73;
    }

    /* ---------------- Canvas ---------------- */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(900px 520px at 12% -8%,
                rgba(124, 92, 255, 0.40), transparent 60%),
            radial-gradient(760px 480px at 92% 4%,
                rgba(45, 225, 194, 0.20), transparent 62%),
            radial-gradient(680px 520px at 70% 100%,
                rgba(255, 93, 115, 0.16), transparent 60%),
            var(--base);
        color: var(--ink);
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"], .stMarkdown, p, li, label, div {
        font-family: 'Inter Tight', system-ui, sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Bricolage Grotesque', 'Inter Tight', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
    }

    .stMarkdown p, .stMarkdown li { color: var(--muted); }

    hr {
        border: none;
        height: 1px;
        background: var(--line);
        margin: 2rem 0;
    }

    /* ---------------- Hero ---------------- */

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 2.6rem 2.6rem 2.2rem;
        background:
            linear-gradient(135deg,
                rgba(124, 92, 255, 0.30) 0%,
                rgba(45, 225, 194, 0.10) 48%,
                rgba(255, 93, 115, 0.22) 100%),
            var(--panel);
        border: 1px solid var(--line);
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        font-size: 13px;
        font-weight: 600;
        color: var(--mint);
        margin-bottom: 18px;
    }

    .pulse {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 0 0 rgba(45, 225, 194, 0.6);
        animation: pulse 2.4s ease-out infinite;
    }

    @keyframes pulse {
        70%  { box-shadow: 0 0 0 11px rgba(45, 225, 194, 0); }
        100% { box-shadow: 0 0 0 0 rgba(45, 225, 194, 0); }
    }

    .hero-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: clamp(38px, 5.4vw, 62px);
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.035em;
        color: #FFFFFF;
        margin-bottom: 14px;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.6;
        color: #D6D0FF;
        max-width: 60ch;
    }

    .spectrum-strip {
        height: 6px;
        border-radius: 99px;
        margin-top: 26px;
        background: linear-gradient(90deg,
            var(--mint), var(--violet), var(--amber), var(--coral));
    }

    .spectrum-legend {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        font-weight: 500;
        color: var(--muted);
        margin-top: 9px;
    }

    /* ---------------- Stat rail ---------------- */

    .rail {
        display: flex;
        flex-wrap: wrap;
        margin: 22px 0 34px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(28, 22, 80, 0.72);
        backdrop-filter: blur(9px);
        overflow: hidden;
    }

    .rail-item {
        flex: 1 1 200px;
        padding: 20px 24px;
        border-right: 1px solid var(--line);
    }

    .rail-item:last-child { border-right: none; }

    .rail-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .rail-value {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }

    .rail-value span { font-size: 16px; color: var(--muted); }

    /* ---------------- Gauge ---------------- */

    .result-shell {
        border-radius: 28px;
        border: 1px solid var(--line);
        background: var(--panel);
        padding: 2.4rem 2rem 2rem;
        text-align: center;
    }

    .gauge {
        position: relative;
        width: 250px;
        height: 250px;
        margin: 4px auto 0;
        border-radius: 50%;
        background: conic-gradient(from 135deg,
            var(--mint)   0deg,
            var(--violet) 88deg,
            var(--amber)  176deg,
            var(--coral)  264deg,
            rgba(255, 255, 255, 0.07) 270deg 360deg);
    }

    .gauge-core {
        position: absolute;
        inset: 26px;
        border-radius: 50%;
        background: var(--panel);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .gauge-number {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 54px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.04em;
        color: #FFFFFF;
    }

    .gauge-caption {
        font-size: 12.5px;
        font-weight: 500;
        color: var(--muted);
        margin-top: 8px;
        max-width: 130px;
        line-height: 1.4;
    }

    .marker {
        position: absolute;
        inset: 0;
        transform: rotate(calc(135deg + (var(--p) * 2.7deg)));
    }

    .marker i {
        position: absolute;
        top: 4px;
        left: 50%;
        margin-left: -9px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #FFFFFF;
        box-shadow:
            0 0 0 5px var(--c),
            0 0 26px 3px var(--c);
    }

    .verdict {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 22px 0 10px;
        color: var(--c);
    }

    .verdict-note {
        font-size: 15px;
        line-height: 1.6;
        color: #D6D0FF;
        max-width: 54ch;
        margin: 0 auto;
    }

    .band-row {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin-top: 24px;
        flex-wrap: wrap;
    }

    .band {
        font-size: 12.5px;
        font-weight: 600;
        padding: 7px 15px;
        border-radius: 99px;
        border: 1px solid var(--line);
        color: var(--muted);
    }

    .band.on {
        color: #12102E;
        border-color: transparent;
    }

    /* ---------------- Callout ---------------- */

    .callout {
        border-radius: 18px;
        padding: 18px 22px;
        font-size: 15px;
        line-height: 1.6;
        color: #EDE9FF;
        border-left: 4px solid var(--c);
        background: rgba(255, 255, 255, 0.045);
    }

    /* ---------------- Form & inputs ---------------- */

    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 2rem 2rem 1.6rem;
        background: rgba(28, 22, 80, 0.62);
        backdrop-filter: blur(10px);
    }

    [data-testid="stWidgetLabel"] p {
        color: #CFC8FF !important;
        font-weight: 600;
        font-size: 13.5px;
    }

    .stNumberInput input,
    .stTextInput input,
    div[data-baseweb="select"] > div {
        background: rgba(12, 9, 38, 0.85) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        color: var(--ink) !important;
        font-weight: 500;
    }

    .stNumberInput input:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--violet) !important;
        box-shadow: 0 0 0 3px rgba(124, 92, 255, 0.28) !important;
    }

    .stNumberInput button {
        background: rgba(124, 92, 255, 0.16) !important;
        border: none !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="popover"] li {
        background: var(--panel-2) !important;
        color: var(--ink) !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        height: 3.4rem;
        border: none;
        border-radius: 14px;
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 16.5px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #0E0B28;
        background: linear-gradient(95deg,
            var(--mint), var(--violet) 58%, var(--coral));
        transition: transform 0.16s ease, filter 0.16s ease;
    }

    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        color: #0E0B28;
    }

    [data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 3px solid var(--mint);
        outline-offset: 3px;
    }

    /* ---------------- Metrics, progress, expander ---------------- */

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 18px;
    }

    [data-testid="stMetricLabel"] p { color: var(--muted) !important; }

    [data-testid="stMetricValue"] {
        font-family: 'Bricolage Grotesque', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
    }

    [data-testid="stProgress"] > div > div > div > div {
        background-image: linear-gradient(90deg,
            var(--mint), var(--violet), var(--amber), var(--coral));
    }

    [data-testid="stExpander"] details {
        background: rgba(28, 22, 80, 0.62);
        border: 1px solid var(--line);
        border-radius: 18px;
    }

    [data-testid="stExpander"] summary { color: var(--ink); }

    /* ---------------- Sidebar ---------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg,
                rgba(124, 92, 255, 0.24), transparent 42%),
            #0E0B28;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .stMarkdown p { color: var(--muted); }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .brand-dot {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        font-size: 19px;
        background: linear-gradient(135deg, var(--mint), var(--violet));
    }

    .brand-name {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }

    .sidebar-fact {
        display: flex;
        justify-content: space-between;
        padding: 9px 0;
        font-size: 13.5px;
        border-bottom: 1px dashed var(--line);
    }

    .sidebar-fact span:first-child { color: var(--muted); }
    .sidebar-fact span:last-child  { color: var(--ink); font-weight: 600; }

    /* ---------------- Section heading ---------------- */

    .section-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin-bottom: 6px;
    }

    .section-sub {
        font-size: 15px;
        color: var(--muted);
        margin-bottom: 18px;
    }

    /* ---------------- Footer ---------------- */

    .foot {
        text-align: center;
        padding: 34px 0 6px;
        color: var(--muted);
        font-size: 13.5px;
        line-height: 1.9;
    }

    .foot b { color: var(--ink); }

    @media (prefers-reduced-motion: reduce) {
        .pulse { animation: none; }
        [data-testid="stFormSubmitButton"] > button { transition: none; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-dot">🩺</div>
            <div class="brand-name">Diabetes AI</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Gradient boosting risk estimation")

    st.success("Model loaded and ready")

    st.markdown("### How it works")

    st.markdown(
        """
        <div class="sidebar-fact">
            <span>Algorithm</span><span>HistGradientBoosting</span>
        </div>
        <div class="sidebar-fact">
            <span>Task</span><span>Binary classification</span>
        </div>
        <div class="sidebar-fact">
            <span>Output</span><span>Probability 0–100%</span>
        </div>
        <div class="sidebar-fact">
            <span>Features</span><span>8 inputs</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Measured performance")

    st.metric("Accuracy", "97.16%")
    st.metric("ROC-AUC", "0.9771")
    st.metric("Precision", "98.08%")

    st.divider()

    st.caption(
        "Built for study and research. This tool estimates risk from "
        "patterns in data — it does not diagnose anyone."
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-eyebrow">
            <span class="pulse"></span> Model online
        </div>

        <div class="hero-title">
            Read the risk<br>before the symptoms.
        </div>

        <div class="hero-subtitle">
            Enter eight everyday clinical and lifestyle values and the model
            places the patient somewhere on the diabetes risk spectrum,
            with the probability behind that placement.
        </div>

        <div class="spectrum-strip"></div>

        <div class="spectrum-legend">
            <span>Low · 0%</span>
            <span>Moderate · 20%</span>
            <span>Elevated · 50%</span>
            <span>High · 100%</span>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# STAT RAIL
# =========================================================

st.markdown(
    """
    <div class="rail">

        <div class="rail-item">
            <div class="rail-label">Model</div>
            <div class="rail-value">HGB <span>classifier</span></div>
        </div>

        <div class="rail-item">
            <div class="rail-label">Accuracy</div>
            <div class="rail-value">97.16<span>%</span></div>
        </div>

        <div class="rail-item">
            <div class="rail-label">ROC-AUC</div>
            <div class="rail-value">0.9771</div>
        </div>

        <div class="rail-item">
            <div class="rail-label">Inputs used</div>
            <div class="rail-value">8 <span>features</span></div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INPUT
# =========================================================

st.markdown(
    """
    <div class="section-title">Patient assessment</div>
    <div class="section-sub">
        Nothing is stored. Values stay in this session only.
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("patient_form"):

    col1, col2 = st.columns(2)

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

    with col2:

        st.markdown("#### Clinical & lifestyle")

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        heart_disease = st.selectbox(
            "Heart disease",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        smoking_history = st.selectbox(
            "Smoking history",
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

    st.markdown("#### Lab results")

    c1, c2 = st.columns(2)

    with c1:

        hba1c = st.number_input(
            "HbA1c level (%)",
            min_value=3.5,
            max_value=9.0,
            value=5.5,
            step=0.1
        )

    with c2:

        glucose = st.number_input(
            "Blood glucose (mg/dL)",
            min_value=80,
            max_value=300,
            value=140,
            step=1
        )

    st.divider()

    predict = st.form_submit_button(
        "Estimate diabetes risk",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict:

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

    if age <= 0:
        st.error("Age must be greater than zero.")

    elif bmi <= 0:
        st.error("BMI must be greater than zero.")

    elif hba1c <= 0:
        st.error("Enter a valid HbA1c value.")

    elif glucose <= 0:
        st.error("Enter a valid blood glucose value.")

    else:

        processed_data = preprocessor.transform(patient_data)

        prediction = model.predict(processed_data)[0]

        probability = float(model.predict_proba(processed_data)[0][1])

        probability_percent = probability * 100

        label, color, message = risk_band(probability)

        headline = (
            "Diabetes risk detected"
            if prediction == 1
            else "No diabetes risk detected"
        )

        # -------------------------------------------------
        # RESULT — spectrum gauge
        # -------------------------------------------------

        st.divider()

        bands = [
            ("Low", "#2DE1C2"),
            ("Moderate", "#7C5CFF"),
            ("Elevated", "#FFB020"),
            ("High", "#FF5D73"),
        ]

        band_chips = "".join(
            f'<div class="band {"on" if name == label else ""}" '
            f'style="{"background:" + hex_code if name == label else ""}">'
            f"{name}</div>"
            for name, hex_code in bands
        )

        st.markdown(
            f"""
            <div class="result-shell" style="--c:{color}">

                <div class="gauge" style="--p:{probability_percent:.2f};
                                          --c:{color}">
                    <div class="marker"><i></i></div>
                    <div class="gauge-core">
                        <div class="gauge-number">
                            {probability_percent:.1f}%
                        </div>
                        <div class="gauge-caption">
                            estimated probability of diabetes
                        </div>
                    </div>
                </div>

                <div class="verdict">{headline}</div>

                <div class="verdict-note">{message}</div>

                <div class="band-row">{band_chips}</div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title" style="margin-top:34px">'
            "What went in</div>",
            unsafe_allow_html=True
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Age", f"{age:.0f}")

        with s2:
            st.metric("BMI", f"{bmi:.1f}")

        with s3:
            st.metric("HbA1c", f"{hba1c:.1f}%")

        with s4:
            st.metric("Glucose", f"{glucose} mg/dL")

        st.progress(min(probability, 1.0))

        st.caption(
            f"Position on the risk spectrum: {probability_percent:.2f}%"
        )

        with st.expander("See the exact values sent to the model"):

            display_data = patient_data.copy()

            display_data["hypertension"] = display_data[
                "hypertension"
            ].map({0: "No", 1: "Yes"})

            display_data["heart_disease"] = display_data[
                "heart_disease"
            ].map({0: "No", 1: "Yes"})

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title" style="margin-top:30px">'
            "Reading the number</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="callout" style="--c:{color}">
                <b>{label} band.</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "A model estimates patterns, not diagnoses. Confirm anything "
            "concerning with a clinician and a lab test."
        )


# =========================================================
# ABOUT MODEL
# =========================================================

st.divider()

with st.expander("How this model was built"):

    st.markdown(
        """
        The predictor is a supervised classifier trained to estimate the
        probability that a person has diabetes, given eight routinely
        collected values.

        **Algorithm** — HistGradientBoostingClassifier, a tree ensemble that
        splits on histogram bins, so it handles the mixed numeric and
        categorical inputs here without much tuning.

        **Inputs** — gender, age, hypertension, heart disease, smoking
        history, BMI, HbA1c level, blood glucose level.

        **Held-out results** — accuracy 97.16%, precision 98.08%,
        F1 score 0.8112, ROC-AUC 0.9771. The gap between the high accuracy
        and the lower F1 is worth noting: the classes are imbalanced, so
        ROC-AUC and precision describe the model better than accuracy does.

        **Preprocessing** — numeric features are scaled and categorical
        features one-hot encoded in a pipeline that runs before every
        prediction, so inputs here are treated exactly as training data was.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="foot">
        <b>Diabetes AI Predictor</b><br>
        Python · scikit-learn · Streamlit<br>
        Educational and research use only
    </div>
    """,
    unsafe_allow_html=True
)
