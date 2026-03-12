import streamlit as st
import requests
import pandas as pd

st.title("AI Retail Supply Intelligence Dashboard")

API_URL = "http://localhost:8000"  # important when running with docker-compose

stores = ["Mumbai","Pune","Nagpur","Nashik","Aurangabad"]
products = ["milk","bread","rice","onion","potato"]

store = st.selectbox("Select Store", stores)
product = st.selectbox("Select Product", products)

# Forecast
if st.button("Get Demand Forecast"):
    response = requests.get(f"{API_URL}/forecast/{store}/{product}")
    data = response.json()

    df = pd.DataFrame(data)

    st.subheader("7-Day Demand Forecast")
    st.dataframe(df)

    if not df.empty:
        st.line_chart(df.set_index("date")["forecast_sales"])

# Restock
if st.button("Get Restock Recommendation"):
    response = requests.get(f"{API_URL}/restock/{store}/{product}")
    data = response.json()

    df = pd.DataFrame(data)

    st.subheader("Restock Recommendation")
    st.dataframe(df)

    if not df.empty:
        st.metric(
            label="Recommended Restock",
            value=int(df["recommended_restock"].values[0])
        )