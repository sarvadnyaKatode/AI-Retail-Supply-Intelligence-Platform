import pandas as pd
from prophet import Prophet
import joblib
import os
import json
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

METRICS_FILE = os.path.join(BASE_DIR, "data", "model_metrics.csv")

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "data", "retail_demand_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

results = []
metrics_log = []
stores = df["store"].unique()
products = df["product"].unique()
total = len(stores) * len(products)
count = 0

print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {total} Prophet models...")

for store in stores:
    for product in products:
        count += 1
        model_path = os.path.join(MODELS_DIR, f"{store}_{product}.pkl")

        data = (
            df[(df["store"] == store) & (df["product"] == product)]
            .groupby("date")["sales"].sum().reset_index()
            .rename(columns={"date": "ds", "sales": "y"})
        )

        if len(data) < 30:
            continue

        print(f"  [{count}/{total}] {store} / {product} ...", end="")

        # ── Walk-forward validation (last 30 days as holdout) ──
        train_data = data.iloc[:-30]
        test_data = data.iloc[-30:]

        # Train final model on full data
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(" (cached)", end="")
        else:
            model = Prophet()
            model.fit(data)
            joblib.dump(model, model_path)

        # Evaluate on holdout using fresh model on train-only data
        eval_model = Prophet()
        eval_model.fit(train_data)
        future_eval = eval_model.make_future_dataframe(periods=30)
        forecast_eval = eval_model.predict(future_eval)
        pred_vals = forecast_eval.iloc[-30:]["yhat"].values
        actual_vals = test_data["y"].values

        mape_val = mape(actual_vals, pred_vals)
        rmse_val = rmse(actual_vals, pred_vals)

        metrics_log.append({
            "store": store,
            "product": product,
            "mape": round(mape_val, 2),
            "rmse": round(rmse_val, 2),
            "n_train": len(train_data),
            "n_test": 30,
        })

        # Generate forecast using cached/trained full model
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        future_forecast = forecast.tail(7)[["ds", "yhat"]]

        for _, row in future_forecast.iterrows():
            results.append({
                "store": store,
                "product": product,
                "date": row["ds"],
                "forecast_sales": max(0, row["yhat"]),
            })

        print(f" MAPE={mape_val:.1f}%  RMSE={rmse_val:.1f}")

# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
forecast_df = pd.DataFrame(results)
forecast_df.to_csv(os.path.join(BASE_DIR, "data", "demand_forecasts.csv"), index=False)

metrics_df = pd.DataFrame(metrics_log)
metrics_df.to_csv(METRICS_FILE, index=False)

avg_mape = metrics_df["mape"].mean()
print(f"\n✅ Forecasts saved: {len(forecast_df)} rows")
print(f"✅ Model metrics saved: {len(metrics_df)} models")
print(f"📊 Overall Average MAPE: {avg_mape:.2f}%")
print(f"💾 Models cached in: {MODELS_DIR}")