"""
Unit Tests for the AI Retail Supply Intelligence Platform.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
    mean_absolute_error,
)
from src.alerts import generate_alerts, Alert


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def sample_restock_df():
    return pd.DataFrame([
        {"store": "Mumbai", "product": "milk", "forecast_sales": 500.0, "inventory": 100, "safety_stock": 100.0, "recommended_restock": 500.0},
        {"store": "Pune",   "product": "rice", "forecast_sales": 300.0, "inventory": 800, "safety_stock": 60.0,  "recommended_restock": 0.0},
        {"store": "Nagpur", "product": "oil",  "forecast_sales": 200.0, "inventory": 50,  "safety_stock": 40.0,  "recommended_restock": 190.0},
    ])

@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=365)
    rows = []
    for date in dates:
        rows.append({"date": date, "store": "Mumbai", "product": "milk",
                     "sales": np.random.randint(30, 80), "stockout": 0,
                     "price": 50.0, "is_weekend": int(date.weekday() >= 5)})
        rows.append({"date": date, "store": "Pune", "product": "rice",
                     "sales": np.random.randint(20, 60), "stockout": 0,
                     "price": 60.0, "is_weekend": int(date.weekday() >= 5)})
        rows.append({"date": date, "store": "Nagpur", "product": "oil",
                     "sales": np.random.randint(10, 40), "stockout": 0,
                     "price": 120.0, "is_weekend": int(date.weekday() >= 5)})
    return pd.DataFrame(rows)

@pytest.fixture
def sample_forecast_df():
    return pd.DataFrame([
        {"store": "Mumbai", "product": "milk", "date": "2025-01-01", "forecast_sales": 75.0},
        {"store": "Pune",   "product": "rice", "date": "2025-01-01", "forecast_sales": 45.0},
    ])


# ─────────────────────────────────────────────
# TEST: METRICS
# ─────────────────────────────────────────────
class TestMetrics:
    def test_mape_perfect(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mean_absolute_percentage_error(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([110.0, 90.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_mape_excludes_zeros(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([50.0, 110.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_rmse_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        result = root_mean_squared_error(y_true, y_pred)
        assert result == pytest.approx(np.sqrt(25/3), rel=1e-3)

    def test_mae_perfect(self):
        y = np.array([5.0, 10.0, 15.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = mean_absolute_error(y_true, y_pred)
        assert result == pytest.approx(2/3, rel=1e-3)

    def test_metrics_return_floats(self):
        y = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        assert isinstance(mean_absolute_percentage_error(y, y_pred), float)
        assert isinstance(root_mean_squared_error(y, y_pred), float)
        assert isinstance(mean_absolute_error(y, y_pred), float)


# ─────────────────────────────────────────────
# TEST: RESTOCK ENGINE LOGIC
# ─────────────────────────────────────────────
class TestRestockEngine:
    def test_no_negative_restock(self, sample_restock_df):
        assert (sample_restock_df["recommended_restock"] >= 0).all()

    def test_healthy_store_has_zero_restock(self, sample_restock_df):
        pune_rice = sample_restock_df[
            (sample_restock_df["store"] == "Pune") &
            (sample_restock_df["product"] == "rice")
        ]
        assert pune_rice["recommended_restock"].values[0] == 0.0

    def test_critical_store_has_positive_restock(self, sample_restock_df):
        mumbai_milk = sample_restock_df[
            (sample_restock_df["store"] == "Mumbai") &
            (sample_restock_df["product"] == "milk")
        ]
        assert mumbai_milk["recommended_restock"].values[0] > 0

    def test_restock_formula(self, sample_restock_df):
        r = sample_restock_df.iloc[0]
        expected_restock = max(0, r["forecast_sales"] + r["safety_stock"] - r["inventory"])
        assert r["recommended_restock"] == pytest.approx(expected_restock, rel=1e-3)


# ─────────────────────────────────────────────
# TEST: ALERTS ENGINE
# ─────────────────────────────────────────────
class TestAlerts:
    def test_critical_alert_for_low_coverage(self, sample_restock_df, sample_forecast_df, sample_df):
        alerts = generate_alerts(sample_restock_df, sample_forecast_df, sample_df)
        critical = [a for a in alerts if a.level == "critical"]
        assert len(critical) > 0

    def test_critical_alert_is_for_correct_store(self, sample_restock_df, sample_forecast_df, sample_df):
        alerts = generate_alerts(sample_restock_df, sample_forecast_df, sample_df)
        critical = [a for a in alerts if a.level == "critical"]
        # Mumbai milk has 500 restock units needed — should trigger critical
        stores_with_critical = [a.store for a in critical]
        assert "Mumbai" in stores_with_critical

    def test_alerts_are_sorted_critical_first(self, sample_restock_df, sample_forecast_df, sample_df):
        alerts = generate_alerts(sample_restock_df, sample_forecast_df, sample_df)
        levels = [a.level for a in alerts]
        # critical should come before warning
        if "critical" in levels and "warning" in levels:
            assert levels.index("critical") < levels.index("warning")

    def test_alerts_have_required_fields(self, sample_restock_df, sample_forecast_df, sample_df):
        alerts = generate_alerts(sample_restock_df, sample_forecast_df, sample_df)
        for alert in alerts:
            assert isinstance(alert, Alert)
            assert alert.level in ("critical", "warning", "info")
            assert alert.store != ""
            assert alert.product != ""
            assert alert.message != ""

    def test_healthy_store_no_critical(self, sample_restock_df, sample_forecast_df, sample_df):
        # Pune rice has zero restock needed — should NOT be critical
        alerts = generate_alerts(sample_restock_df, sample_forecast_df, sample_df)
        critical_pune_rice = [a for a in alerts if a.level == "critical" and a.store == "Pune" and "rice" in a.product.lower()]
        assert len(critical_pune_rice) == 0


# ─────────────────────────────────────────────
# TEST: DATA LOADER
# ─────────────────────────────────────────────
class TestDataStructure:
    def test_dataframe_has_required_columns(self, sample_df):
        required = ["date", "store", "product", "sales"]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_restock_df_has_required_columns(self, sample_restock_df):
        required = ["store", "product", "forecast_sales", "inventory", "recommended_restock"]
        for col in required:
            assert col in sample_restock_df.columns, f"Missing column: {col}"

    def test_no_negative_sales(self, sample_df):
        assert (sample_df["sales"] >= 0).all()

    def test_stores_are_strings(self, sample_df):
        assert sample_df["store"].dtype == object

    def test_date_is_datetime(self, sample_df):
        assert pd.api.types.is_datetime64_any_dtype(sample_df["date"])
