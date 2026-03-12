import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data/retail_demand_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

# example: forecast demand for milk in Mumbai
data = df[(df["product"] == "milk") & (df["store"] == "Mumbai")]

# aggregate daily sales
data = data.groupby("date")["sales"].sum().reset_index()

# prophet format
data = data.rename(columns={"date": "ds", "sales": "y"})

# create model
model = Prophet()

model.fit(data)

# forecast next 7 days
future = model.make_future_dataframe(periods=7)

forecast = model.predict(future)

print(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(7))

# plot forecast
fig = model.plot(forecast)

plt.savefig("docs/demand_forecast.png")

plt.show()