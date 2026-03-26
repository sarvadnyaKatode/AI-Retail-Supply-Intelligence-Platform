from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import Optional, List

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(
    title="AI Retail Supply Intelligence API",
    description="AI-powered demand forecasting and inventory optimization for Maharashtra retail network.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found. Run the forecasting pipeline first.")
    return pd.read_csv(path)

try:
    forecast_df = load_csv("demand_forecasts.csv")
    restock_df  = load_csv("restock_recommendations.csv")
    metrics_df  = load_csv("model_metrics.csv") if os.path.exists(os.path.join(DATA_DIR, "model_metrics.csv")) else pd.DataFrame()
except FileNotFoundError as e:
    import warnings
    warnings.warn(str(e))
    forecast_df = pd.DataFrame()
    restock_df  = pd.DataFrame()
    metrics_df  = pd.DataFrame()

VALID_STORES   = sorted(forecast_df["store"].unique().tolist())   if not forecast_df.empty else []
VALID_PRODUCTS = sorted(forecast_df["product"].unique().tolist()) if not forecast_df.empty else []

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def validate_store_product(store: str, product: str):
    if VALID_STORES and store not in VALID_STORES:
        raise HTTPException(status_code=404, detail=f"Store '{store}' not found. Valid: {VALID_STORES}")
    if VALID_PRODUCTS and product not in VALID_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product '{product}' not found. Valid: {VALID_PRODUCTS}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def home():
    return {
        "status": "healthy",
        "message": "AI Retail Supply Intelligence API v2.0",
        "stores": len(VALID_STORES),
        "products": len(VALID_PRODUCTS),
        "docs": "/docs",
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "forecast_rows": len(forecast_df),
        "restock_rows": len(restock_df),
        "models_evaluated": len(metrics_df),
    }

@app.get("/stores", tags=["Reference"])
def list_stores():
    """List all available store names."""
    return {"stores": VALID_STORES}

@app.get("/products", tags=["Reference"])
def list_products():
    """List all available product names."""
    return {"products": VALID_PRODUCTS}

@app.get("/forecast/{store}/{product}", tags=["Forecast"])
def get_forecast(store: str, product: str):
    """Get 7-day demand forecast for a specific store and product."""
    validate_store_product(store, product)
    result = forecast_df[
        (forecast_df["store"] == store) &
        (forecast_df["product"] == product)
    ]
    if result.empty:
        raise HTTPException(status_code=404, detail="No forecast data found for this combination.")
    return {
        "store": store,
        "product": product,
        "forecast": result[["date", "forecast_sales"]].to_dict(orient="records"),
        "total_7day": round(result["forecast_sales"].sum(), 1),
        "avg_daily": round(result["forecast_sales"].mean(), 1),
    }

@app.get("/restock/{store}/{product}", tags=["Restock"])
def get_restock(store: str, product: str):
    """Get restock recommendation for a specific store and product."""
    validate_store_product(store, product)
    result = restock_df[
        (restock_df["store"] == store) &
        (restock_df["product"] == product)
    ]
    if result.empty:
        raise HTTPException(status_code=404, detail="No restock data found for this combination.")
    r = result.iloc[0]
    return {
        "store": store,
        "product": product,
        "forecast_demand_7d": round(float(r["forecast_sales"]), 1),
        "current_inventory": int(r["inventory"]),
        "safety_stock": round(float(r["safety_stock"]), 1),
        "recommended_restock": int(r["recommended_restock"]),
        "status": "CRITICAL" if r["recommended_restock"] > 200 else ("NEEDED" if r["recommended_restock"] > 0 else "OK"),
    }

@app.get("/alerts", tags=["Intelligence"])
def get_alerts(store: Optional[str] = Query(None, description="Filter by store")):
    """Get smart alerts: stockout risks, demand spikes, and overstock warnings."""
    if restock_df.empty:
        return {"alerts": []}
    
    df = restock_df.copy()
    if store:
        if store not in VALID_STORES:
            raise HTTPException(status_code=404, detail=f"Store '{store}' not found.")
        df = df[df["store"] == store]

    alerts = []
    for _, row in df.iterrows():
        coverage_pct = (row["inventory"] / max(row["forecast_sales"], 1)) * 100
        if coverage_pct < 50 and row["recommended_restock"] > 0:
            alerts.append({
                "level": "CRITICAL",
                "store": row["store"],
                "product": row["product"],
                "message": f"Only {coverage_pct:.0f}% inventory coverage — restock {int(row['recommended_restock'])} units",
                "restock_units": int(row["recommended_restock"]),
            })
        elif row["inventory"] > row["forecast_sales"] * 2:
            alerts.append({
                "level": "WARNING",
                "store": row["store"],
                "product": row["product"],
                "message": f"Overstock: {int(row['inventory'])} units vs {int(row['forecast_sales'])} forecast",
                "excess_units": int(row["inventory"] - row["forecast_sales"]),
            })
    
    alerts.sort(key=lambda x: 0 if x["level"] == "CRITICAL" else 1)
    return {"total": len(alerts), "alerts": alerts}

@app.get("/metrics", tags=["Model Performance"])
def get_model_metrics(store: Optional[str] = None, product: Optional[str] = None):
    """Get Prophet model accuracy metrics (MAPE, RMSE). Run forecast_all_products.py to generate."""
    if metrics_df.empty:
        return {"message": "No metrics available. Run models/forecast_all_products.py first.", "metrics": []}
    
    df = metrics_df.copy()
    if store:
        df = df[df["store"] == store]
    if product:
        df = df[df["product"] == product]
    
    return {
        "avg_mape": round(df["mape"].mean(), 2) if not df.empty else None,
        "avg_rmse": round(df["rmse"].mean(), 2) if not df.empty else None,
        "n_models": len(df),
        "metrics": df.to_dict(orient="records"),
    }

@app.get("/summary", tags=["Intelligence"])
def get_summary():
    """Get a full platform summary: top restock needs, overall stats."""
    if restock_df.empty:
        return {"error": "No data available"}
    
    top_restock = restock_df.sort_values("recommended_restock", ascending=False).head(5)
    
    return {
        "total_stores": len(VALID_STORES),
        "total_products": len(VALID_PRODUCTS),
        "total_restock_units": int(restock_df["recommended_restock"].sum()),
        "critical_items": int((restock_df["recommended_restock"] > 200).sum()),
        "top_5_restock_needs": top_restock[["store","product","recommended_restock"]].to_dict(orient="records"),
    }