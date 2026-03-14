import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Crop Price Predictor",
    page_icon="🌾",
    layout="wide"
)
st.title("🌾 Crop Price Prediction Dashboard")
st.caption("Machine Learning powered agricultural price estimation")

st.divider()
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "price_model.pkl"
model = joblib.load(model_path)
data_path = BASE_DIR / "data" / "crop_prices.csv"
df = pd.read_csv(data_path)
st.sidebar.header("Select Market Information")
state = st.sidebar.selectbox(
    "State",
    sorted(df["State"].unique())
)

commodity = st.sidebar.selectbox(
    "Commodity",
    sorted(df["Commodity"].unique())
)

market = st.sidebar.selectbox(
    "Market",
    sorted(df["Market"].unique())
)
st.subheader("Enter Market Prices")

col1, col2 = st.columns(2)

with col1:
    min_price = st.number_input(
        "Minimum Price (₹)",
        min_value=0.0,
        step=10.0
    )

with col2:
    max_price = st.number_input(
        "Maximum Price (₹)",
        min_value=0.0,
        step=10.0
    )
st.divider()
predict_button = st.button("Predict Modal Price")
if predict_button:

    if max_price < min_price:
        st.error("Maximum price must be greater than minimum price.")
      
    else:    
        input_df = pd.DataFrame({
            "State": [state],
            "Market": [market],
            "Commodity": [commodity],
            "Min_x0020_Price": [min_price],
            "Max_x0020_Price": [max_price]
        })

        input_df = pd.get_dummies(input_df)
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)      
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Modal Price",
            value=f"₹{prediction:,.2f}"
        )
        st.success("Prediction generated successfully.")
st.divider()
st.subheader("Dataset Price Distribution")
st.line_chart(df["Modal_x0020_Price"])
