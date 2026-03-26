"""
Smart Alerts Engine for the Retail Intelligence Platform.

Alert Types:
  - CRITICAL: Inventory coverage < 50% of 7-day forecast
  - WARNING:  Demand spike (forecast > 1.35x historical average)
  - WARNING:  Overstock (inventory > 2x forecast)
  - INFO:     Healthy inventory
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class Alert:
    level: str        # "critical" | "warning" | "info"
    store: str
    product: str
    message: str
    value: float = 0.0
    icon: str = "ℹ️"


def generate_alerts(
    restock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    df: pd.DataFrame,
    max_alerts: int = 20,
) -> List[Alert]:
    """
    Generate smart alerts based on restock recommendations and historical data.
    
    Args:
        restock_df: Output from restock engine (store, product, forecast_sales, inventory, recommended_restock)
        forecast_df: 7-day demand forecasts
        df: Full historical retail DataFrame
        max_alerts: Maximum alerts to return
    
    Returns:
        List of Alert objects sorted by severity
    """
    alerts: List[Alert] = []

    # Precompute historical daily averages
    hist_avg = (
        df.groupby(["store", "product"])["sales"]
        .mean()
        .rename("hist_avg_daily")
        .reset_index()
    )
    merged = restock_df.merge(hist_avg, on=["store", "product"], how="left")

    for _, row in merged.iterrows():
        store, product = row["store"], row["product"]
        forecast = row["forecast_sales"]
        inventory = row["inventory"]
        restock = row["recommended_restock"]
        hist_7day = row.get("hist_avg_daily", 0) * 7

        # Coverage ratio: how many days of inventory we have
        coverage_pct = (inventory / max(forecast, 1)) * 100

        # ── CRITICAL: near stockout ──
        if restock > 0 and coverage_pct < 50:
            alerts.append(Alert(
                level="critical",
                store=store,
                product=product.title(),
                message=f"{store} — {product.title()}: Only {coverage_pct:.0f}% inventory coverage. Needs **{int(restock)} units** restocked ASAP.",
                value=restock,
                icon="🚨",
            ))

        # ── WARNING: demand spike ──
        elif hist_7day > 0 and forecast > hist_7day * 1.35:
            spike_pct = ((forecast / hist_7day) - 1) * 100
            alerts.append(Alert(
                level="warning",
                store=store,
                product=product.title(),
                message=f"{store} — {product.title()}: Demand forecast is **{spike_pct:.0f}% above** historical average this week.",
                value=spike_pct,
                icon="📈",
            ))

        # ── WARNING: overstock ──
        elif inventory > forecast * 2 and forecast > 0:
            excess = inventory - forecast
            alerts.append(Alert(
                level="warning",
                store=store,
                product=product.title(),
                message=f"{store} — {product.title()}: Overstock detected. Current inventory is **{inventory:.0f}** vs forecast demand **{forecast:.0f}**. Risk of waste.",
                value=excess,
                icon="📦",
            ))

    # Sort: critical first, then warning, then info
    level_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (level_order.get(a.level, 3), -a.value))

    return alerts[:max_alerts]


def get_store_alerts(
    store: str,
    restock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    df: pd.DataFrame,
) -> List[Alert]:
    """Get alerts filtered to a specific store."""
    all_alerts = generate_alerts(restock_df, forecast_df, df)
    return [a for a in all_alerts if a.store == store]
