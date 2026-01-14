import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LinearRegression




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


X = df[
    [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership"
    ]
]

y = df["Yearly Amount Spent"]


# TRAIN MODEL

model = LinearRegression()
model.fit(X, y)

# ===============================
# STREAMLIT UI
# ===============================
st.title("E-Commerce Customer Spending Prediction")

st.sidebar.title("Input Customer Data")

avg_session = st.sidebar.slider(
    "Avg. Session Length",
    float(df["Avg. Session Length"].min()),
    float(df["Avg. Session Length"].max())
)

time_app = st.sidebar.slider(
    "Time on App",
    float(df["Time on App"].min()),
    float(df["Time on App"].max())
)

time_web = st.sidebar.slider(
    "Time on Website",
    float(df["Time on Website"].min()),
    float(df["Time on Website"].max())
)

membership = st.sidebar.slider(
    "Length of Membership",
    float(df["Length of Membership"].min()),
    float(df["Length of Membership"].max())
)

# ===============================
# PREDICTION
# ===============================
input_data = [[avg_session, time_app, time_web, membership]]
prediction = model.predict(input_data)

# ===============================
# OUTPUT
# ===============================
st.subheader("Prediction Result")
st.write(f"💰 **Predicted Yearly Amount Spent:** ${prediction[0]:.2f}")
