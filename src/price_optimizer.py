"""
Price Optimization Engine
Uses price elasticity from historical data to recommend optimal discounts
that maximize revenue for each store-product combination.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def estimate_price_elasticity(df: pd.DataFrame, store: str, product: str) -> Optional[float]:
    """
    Estimate price elasticity of demand for a store-product pair.
    Elasticity = % change in quantity / % change in price

    Args:
        df: Full retail DataFrame
        store: Store name
        product: Product name
    
    Returns:
        Elasticity coefficient (typically negative; -1 = unit elastic)
    """
    data = df[(df["store"] == store) & (df["product"] == product)][["price", "sales"]].copy()
    if len(data) < 30:
        return None

    # Log-log regression: log(sales) ~ log(price) gives elasticity directly
    data = data[data["price"] > 0]
    log_price = np.log(data["price"].values)
    log_sales = np.log(data["sales"].values + 1)

    # OLS slope
    n = len(log_price)
    x_mean, y_mean = log_price.mean(), log_sales.mean()
    numerator = np.sum((log_price - x_mean) * (log_sales - y_mean))
    denominator = np.sum((log_price - x_mean) ** 2)

    if denominator == 0:
        return None
    return float(numerator / denominator)


def optimal_discount(
    base_price: float,
    elasticity: float,
    current_units: float,
    margin_pct: float = 0.25,
    max_discount: float = 0.30,
) -> dict:
    """
    Find the discount that maximizes profit given price elasticity.

    Formula:
        Revenue(d) = price*(1-d) * units*(1 + elasticity * d)
        Profit(d) = Revenue(d) * margin_pct

    Args:
        base_price: Current unit price
        elasticity: Price elasticity (negative for normal goods)
        current_units: Current average daily units sold
        margin_pct: Gross margin percentage
        max_discount: Maximum discount fraction (0.30 = 30%)
    
    Returns:
        dict with optimal_discount_pct, optimal_price, projected_units, projected_revenue, revenue_lift_pct
    """
    discounts = np.linspace(0, max_discount, 100)
    best = {"profit": -np.inf, "discount": 0.0}

    for d in discounts:
        price = base_price * (1 - d)
        units = current_units * (1 + abs(elasticity) * d)  # demand increases as price drops
        revenue = price * units
        profit = revenue * margin_pct
        if profit > best["profit"]:
            best = {"profit": profit, "discount": d, "price": price, "units": units, "revenue": revenue}

    base_revenue = base_price * current_units * margin_pct
    lift_pct = ((best["profit"] - base_revenue) / max(base_revenue, 1)) * 100

    return {
        "optimal_discount_pct": round(best["discount"] * 100, 1),
        "optimal_price": round(best["price"], 2),
        "projected_units": round(best["units"], 1),
        "projected_daily_revenue": round(best["revenue"], 2),
        "profit_lift_pct": round(lift_pct, 1),
    }


def price_recommendations_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price optimization recommendations for all store-product combos.
    
    Returns:
        DataFrame with store, product, elasticity, optimal_discount_pct, profit_lift_pct
    """
    results = []
    avg_prices = df.groupby("product")["price"].mean()
    avg_units = df.groupby(["store", "product"])["sales"].mean()

    for store in df["store"].unique():
        for product in df["product"].unique():
            elasticity = estimate_price_elasticity(df, store, product)
            if elasticity is None or elasticity >= 0:
                continue

            base_price = avg_prices.get(product, 50)
            current_units = avg_units.get((store, product), 30)

            rec = optimal_discount(
                base_price=base_price,
                elasticity=elasticity,
                current_units=current_units,
            )
            results.append({
                "store": store,
                "product": product,
                "elasticity": round(elasticity, 3),
                **rec,
            })

    return pd.DataFrame(results).sort_values("profit_lift_pct", ascending=False)
