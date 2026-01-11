import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Spending Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    * {
        font-weight: 600 !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.2rem;
        padding: 0;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
    }
    h2, h3, h1 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-weight: bold !important;
    }
    .streamlit-expanderHeader {
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💰 Spending Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer spending based on engagement metrics</div>', unsafe_allow_html=True)

# LOAD DATA


@st.cache_data
def load_data():
    base = Path(__file__).parent
    candidates = [
        base / "Ecommerce_Customers.csv",
        base / "Ecommerce Customers.csv",
        base / "Ecommerce Customers",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    # try any file starting with Ecommerce in this folder
    for f in base.glob("Ecommerce*"):
        if f.is_file():
            return pd.read_csv(f)
    raise FileNotFoundError(
        f"Dataset not found in {base}. Place the CSV file 'Ecommerce_Customers.csv' in that folder."
    )


df = load_data()

# FEATURES & TARGET
X = df[[
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership"
]]

y = df["Yearly Amount Spent"]

# TRAIN MODEL
model = LinearRegression()
model.fit(X, y)

# STREAMLIT UI

st.title("🎯 Spending Prediction")

st.sidebar.header("📊 Input Data")

avg_session = st.sidebar.slider(
    "⏱️ Session Length",
    float(df["Avg. Session Length"].min()),
    float(df["Avg. Session Length"].max()),
    value=float(df["Avg. Session Length"].mean())
)

time_app = st.sidebar.slider(
    "📱 App Time",
    float(df["Time on App"].min()),
    float(df["Time on App"].max()),
    value=float(df["Time on App"].mean())
)

time_web = st.sidebar.slider(
    "🌐 Website Time",
    float(df["Time on Website"].min()),
    float(df["Time on Website"].max()),
    value=float(df["Time on Website"].mean())
)

membership = st.sidebar.slider(
    "📅 Membership",
    float(df["Length of Membership"].min()),
    float(df["Length of Membership"].max()),
    value=float(df["Length of Membership"].mean())
)

# PREDICTION
input_data = [[avg_session, time_app, time_web, membership]]
prediction = model.predict(input_data)

# OUTPUT
st.markdown("### 💵 Prediction")
st.markdown(f"""
    <div style="background-color: #d4edda; padding: 1.2rem; border-radius: 0.5rem; border-left: 4px solid #28a745;">
        <h3 style="color: #155724; margin: 0; font-weight: bold;">💰 ${prediction[0]:,.2f}</h3>
        <p style="color: #155724; font-size: 0.9rem; margin: 0.3rem 0 0 0; font-weight: 600;">Yearly Spending</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Input Summary")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Session", f"{avg_session:.0f}m", delta=None)
with col_b:
    st.metric("App", f"{time_app:.0f}m", delta=None)
with col_c:
    st.metric("Web", f"{time_web:.0f}m", delta=None)
