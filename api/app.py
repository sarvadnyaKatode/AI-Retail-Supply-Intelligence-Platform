from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="AI Retail Supply Intelligence API")

# load forecast and restock data
forecast_df = pd.read_csv("data/demand_forecasts.csv")
restock_df = pd.read_csv("data/restock_recommendations.csv")


@app.get("/")
def home():
    return {"message": "Retail Demand Intelligence API Running"}


# -----------------------------
# Forecast Endpoint
# -----------------------------

@app.get("/forecast/{store}/{product}")
def get_forecast(store: str, product: str):

    result = forecast_df[
        (forecast_df["store"] == store) &
        (forecast_df["product"] == product)
    ]

    return result.to_dict(orient="records")


# -----------------------------
# Restock Endpoint
# -----------------------------

@app.get("/restock/{store}/{product}")
def get_restock(store: str, product: str):

    result = restock_df[
        (restock_df["store"] == store) &
        (restock_df["product"] == product)
    ]

    return result.to_dict(orient="records")