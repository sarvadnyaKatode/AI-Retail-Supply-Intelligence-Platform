"""Data loading utilities for the Retail Intelligence Platform."""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_retail_data(path: str = None) -> pd.DataFrame:
    """Load and preprocess the main retail dataset."""
    path = path or os.path.join(DATA_DIR, "retail_demand_dataset.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_forecasts(path: str = None) -> pd.DataFrame:
    """Load pre-computed demand forecasts."""
    path = path or os.path.join(DATA_DIR, "demand_forecasts.csv")
    return pd.read_csv(path, parse_dates=["date"])


def load_restock(path: str = None) -> pd.DataFrame:
    """Load pre-computed restock recommendations."""
    path = path or os.path.join(DATA_DIR, "restock_recommendations.csv")
    return pd.read_csv(path)


def load_metrics(path: str = None) -> pd.DataFrame:
    """Load model accuracy metrics if available."""
    path = path or os.path.join(DATA_DIR, "model_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def get_store_product_data(df: pd.DataFrame, store: str, product: str) -> pd.DataFrame:
    """Filter dataset to a single store-product combination and aggregate by date."""
    return (
        df[(df["store"] == store) & (df["product"] == product)]
        .groupby("date")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "sales": "y"})
    )
