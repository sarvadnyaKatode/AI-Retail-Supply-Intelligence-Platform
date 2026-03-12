import pandas as pd

# load forecast results
forecast = pd.read_csv("data/demand_forecasts.csv")

# load original dataset
df = pd.read_csv("data/retail_demand_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

# get latest inventory per store/product
latest_inventory = (
    df.sort_values("date")
    .groupby(["store","product"])
    .tail(1)[["store","product","inventory"]]
)

# aggregate forecast demand for next 7 days
forecast_7day = (
    forecast.groupby(["store","product"])["forecast_sales"]
    .sum()
    .reset_index()
)

# merge inventory with forecast
restock_df = pd.merge(
    forecast_7day,
    latest_inventory,
    on=["store","product"]
)

# safety stock = 20% of forecast
restock_df["safety_stock"] = restock_df["forecast_sales"] * 0.2

# restock calculation
restock_df["recommended_restock"] = (
    restock_df["forecast_sales"]
    + restock_df["safety_stock"]
    - restock_df["inventory"]
)

# avoid negative restock
restock_df["recommended_restock"] = restock_df["recommended_restock"].clip(lower=0)

# round values
restock_df["recommended_restock"] = restock_df["recommended_restock"].round()

# save result
restock_df.to_csv("data/restock_recommendations.csv", index=False)

print("Restock recommendations generated")

print(restock_df.head())