import pandas as pd
from prophet import Prophet

df = pd.read_csv("data/retail_demand_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

results = []

for store in df["store"].unique():
    for product in df["product"].unique():

        data = df[(df["store"]==store) & (df["product"]==product)]

        data = data.groupby("date")["sales"].sum().reset_index()

        data = data.rename(columns={"date":"ds","sales":"y"})

        if len(data) < 30:
            continue

        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        future_forecast = forecast.tail(7)[["ds","yhat"]]

        for _,row in future_forecast.iterrows():
            results.append({
                "store":store,
                "product":product,
                "date":row["ds"],
                "forecast_sales":row["yhat"]
            })

forecast_df = pd.DataFrame(results)

forecast_df.to_csv("data/demand_forecasts.csv",index=False)

print("Forecasts generated")