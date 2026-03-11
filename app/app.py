import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
import joblib

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #080808; color: #e0e0e0; }
    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem; letter-spacing: 0.08em;
        text-align: center; color: #ffffff;
        margin: 1rem 0 0.3rem; line-height: 1;
    }
    .main-title span { color: #e02020; }
    .subtitle {
        text-align: center; color: #888; font-size: 1rem;
        letter-spacing: 0.06em; text-transform: uppercase;
        margin-bottom: 2.5rem; font-weight: 500;
    }
    .result-box {
        background: #0e0e0e; border-radius: 4px;
        border: 1px solid #1e1e1e; border-top: 4px solid #e02020;
        padding: 2.6rem 1.8rem; text-align: center; color: white;
        margin: 2.5rem 0; box-shadow: 0 0 60px rgba(224,32,32,0.12);
    }
    .result-amount {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4.5rem; letter-spacing: 0.05em;
        color: #e02020; margin: 0.5rem 0;
    }
    .result-label {
        font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: #666; font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background: #0a0a0a !important;
        border-right: 1px solid #1a1a1a;
    }
    .sidebar-title {
        font-family: 'Bebas Neue', sans-serif; color: #ffffff;
        font-size: 1.5rem; letter-spacing: 0.1em;
        margin: 1.4rem 0 0.3rem; text-align: center;
    }
    .sidebar-title span { color: #e02020; }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e02020, transparent);
        margin: 1rem 0 1.6rem; opacity: 0.4;
    }
    .stButton > button {
        background: #e02020; color: #ffffff; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        font-size: 0.85rem; border: none; border-radius: 3px;
        padding: 0.85rem 2rem; transition: all 0.22s ease; width: 100%;
    }
    .stButton > button:hover {
        background: #c01818;
        box-shadow: 0 8px 30px rgba(224,32,32,0.35);
        transform: translateY(-2px);
    }
    .stSelectbox > div > div, .stNumberInput > div > div, [data-baseweb="input"] {
        background: #111111 !important; border: 1px solid #2a2a2a !important;
        border-radius: 3px !important; color: #e0e0e0 !important;
    }
    .stSelectbox > div > div:hover, [data-baseweb="input"]:hover { border-color: #e02020 !important; }
    .stRadio > div {
        background: #111111; border: 1px solid #2a2a2a;
        border-radius: 3px; padding: 0.7rem 1rem;
    }
    .stRadio > div:hover { border-color: #e02020; }
    .stSlider > div > div > div > div { background: #e02020 !important; }
    .stNumberInput button { display: none !important; }
    .stNumberInput input {
        background: transparent !important; color: #ffffff !important;
        font-weight: 600 !important; border: none !important;
    }
    .section-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.14em; color: #e02020;
        font-weight: 700; margin-bottom: 1rem; margin-top: 0.2rem;
    }
    hr { border: none; border-top: 1px solid #1e1e1e; margin: 2.5rem 0; }
    .team-card {
        background: #111111; border-radius: 4px; padding: 2rem;
        border: 1px solid #1e1e1e; border-top: 3px solid #e02020;
        margin-bottom: 1rem; transition: all 0.25s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .team-card:hover {
        box-shadow: 0 10px 40px rgba(224,32,32,0.15);
        transform: translateY(-3px); border-top-color: #ff4444;
    }
    .team-card h2 {
        font-family: 'Bebas Neue', sans-serif; color: #ffffff;
        font-size: 2rem; letter-spacing: 0.06em; margin: 0 0 0.3rem 0;
    }
    .team-card .role {
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.14em; color: #e02020;
        font-weight: 700; margin-bottom: 1.2rem;
    }
    .team-card p { color: #999; font-size: 0.95rem; line-height: 1.7; margin: 0; }
    .accent-tag {
        display: inline-block; background: rgba(224,32,32,0.1);
        border: 1px solid rgba(224,32,32,0.25); color: #e02020;
        font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
        padding: 0.2rem 0.6rem; border-radius: 2px;
        margin-right: 0.4rem; margin-bottom: 0.4rem; font-weight: 600;
    }
    .footer {
        text-align: center; color: #333; padding: 2rem 0;
        font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)


# ─── Model + Encoder loading ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path    = "model/customer_churn_model.pkl"
    encoder_path  = "model/encoders.pkl"

    # Load model
    try:
        obj = joblib.load(model_path)
    except Exception:
        try:
            with open(model_path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None, None, None

    model         = obj.get('model')
    feature_names = obj.get('features_name')

    # Load encoders
    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
    except Exception as e:
        st.error(f"Encoder loading failed: {str(e)}")
        return None, None, None

    return model, feature_names, encoders


model, feature_names, encoders = load_artifacts()

if model is None:
    st.warning("Artifacts could not be loaded. Prediction features are disabled.")


# ─── Prediction logic ────────────────────────────────────────────────────────
def predict_churn(input_dict):
    if model is None:
        return None, None
    try:
        df = pd.DataFrame([input_dict])

        # Apply the same LabelEncoder for each categorical column
        for col, enc in encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])

        # Reorder columns to match training order
        if feature_names:
            df = df[feature_names]

        prediction = model.predict(df)[0]
        proba      = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else None
        return int(prediction), proba

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return None, None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">CHURN <span>INTEL</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Make Prediction", "Recent Predictions", "Project Info", "About"],
        label_visibility="collapsed"
    )

if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []


def render_header(subtitle="AI-powered customer churn prediction"):
    st.markdown('<h1 class="main-title">CHURN <span>INTEL</span></h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Make Prediction
# ═══════════════════════════════════════════════════════════════════════════
if page == "Make Prediction":
    render_header()

    if model is None:
        st.warning("Prediction is currently unavailable — artifacts could not be loaded.")
    else:
        with st.form("churn_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="section-label">Customer Info</div>', unsafe_allow_html=True)
                gender         = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner        = st.selectbox("Partner", ["Yes", "No"])
                dependents     = st.selectbox("Dependents", ["Yes", "No"])
                tenure         = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)

            with col2:
                st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
                phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection= st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            with col3:
                st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
                contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing= st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method   = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges  = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
                total_charges    = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0, step=10.0)

            submitted = st.form_submit_button("Run Churn Analysis", use_container_width=True)

        if submitted:
            input_data = {
                "gender":           gender,
                "SeniorCitizen":    senior_citizen,
                "Partner":          partner,
                "Dependents":       dependents,
                "tenure":           tenure,
                "PhoneService":     phone_service,
                "MultipleLines":    multiple_lines,
                "InternetService":  internet_service,
                "OnlineSecurity":   online_security,
                "OnlineBackup":     online_backup,
                "DeviceProtection": device_protection,
                "TechSupport":      tech_support,
                "StreamingTV":      streaming_tv,
                "StreamingMovies":  streaming_movies,
                "Contract":         contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod":    payment_method,
                "MonthlyCharges":   monthly_charges,
                "TotalCharges":     total_charges,
            }

            with st.spinner("Analyzing..."):
                result, proba = predict_churn(input_data)

            if result is not None:
                churn_label = "LIKELY TO CHURN" if result == 1 else "LIKELY TO RETAIN"
                churn_color = "#e02020" if result == 1 else "#22c55e"
                confidence  = f"{max(proba)*100:.1f}%" if proba is not None else "N/A"

                st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Prediction Result</div>
                        <div class="result-amount" style="color:{churn_color};">{churn_label}</div>
                        <div class="result-label" style="margin-top:0.8rem;">Model Confidence &nbsp;|&nbsp; {confidence}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.session_state.recent_predictions.append({
                    "Gender":       gender,
                    "Senior":       "Yes" if senior_citizen else "No",
                    "Tenure (mo)":  tenure,
                    "Contract":     contract,
                    "Monthly ($)":  f"${monthly_charges:,.2f}",
                    "Internet":     internet_service,
                    "Result":       churn_label,
                    "Confidence":   confidence,
                })

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Recent Predictions
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Recent Predictions":
    render_header("Session prediction history")

    if st.session_state.recent_predictions:
        df = pd.DataFrame(st.session_state.recent_predictions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions have been made in this session yet.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Project Info
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Project Info":
    render_header("Technical overview")

    st.markdown("""
**Churn Intel** is a machine learning application designed to predict the probability of customer churn
based on behavioral, contractual, and demographic signals.

**Model:** Random Forest Classifier

**Preprocessing:**
- Categorical columns encoded using LabelEncoder (same encoders applied at inference)
- Features reordered to match training column order

**Input Features:**
- Customer demographics: Gender, Senior Citizen, Partner, Dependents
- Account details: Tenure, Contract type, Paperless Billing, Payment method
- Services: Phone, Internet, Online Security, Backup, Device Protection, Tech Support, Streaming
- Billing: Monthly charges, Total charges

**Disclaimer:** This tool provides probabilistic estimates. Actual churn outcomes depend on factors
beyond the scope of this model, including customer sentiment, support interactions, and competitor activity.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════════════════════════════════
elif page == "About":
    render_header("The developer")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('''
        <div class="team-card">
            <h2>Haider Haroon</h2>
            <div class="role">AI Engineer</div>
            <p>
                Haider is an AI Engineer with a focus on AI Security and Computer Vision.
                He specializes in building robust, production-ready AI systems that address
                real-world challenges across diverse industries. His work spans the full
                development lifecycle — from research and prototyping through to deployment
                and evaluation — with a strong emphasis on reliability and security.
            </p>
            <br>
            <span class="accent-tag">AI Security</span>
            <span class="accent-tag">Computer Vision</span>
            <span class="accent-tag">MLOps</span>
            <span class="accent-tag">Deep Learning</span>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="team-card" style="border-top-color: #333;">
            <h2 style="color: #555;">Connect</h2>
            <div class="role" style="color:#555;">Links & Contact</div>
            <p style="color:#555; font-size:0.9rem; line-height:2.2;">
                GitHub &nbsp;&mdash;&nbsp; github.com/Haid3rH<br>
                Email &nbsp;&mdash;&nbsp; haiderharoon2005@gmail.com
            </p>
        </div>
        ''', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="footer">Built by Haider Haroon &nbsp;|&nbsp; 2026</div>', unsafe_allow_html=True)