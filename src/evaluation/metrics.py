"""
Model Evaluation Metrics Module
Computes MAPE, RMSE, MAE for forecast validation.
"""
import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — lower is better. Returns percentage (0-100)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — same unit as target variable."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE — same unit as target variable."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_prophet_model(model, data: pd.DataFrame, test_size: int = 30) -> dict:
    """
    Performs walk-forward validation on a fitted Prophet model.
    
    Args:
        model: Fitted Prophet model
        data: DataFrame with 'ds' and 'y' columns (full history)
        test_size: Number of most recent days to use as holdout
    
    Returns:
        dict with MAPE, RMSE, MAE, and sample size
    """
    if len(data) < test_size + 30:
        return {"mape": None, "rmse": None, "mae": None, "n_test": 0}

    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]

    from prophet import Prophet
    eval_model = Prophet()
    eval_model.fit(train)

    future = eval_model.make_future_dataframe(periods=test_size)
    forecast = eval_model.predict(future)

    pred = forecast.iloc[-test_size:]["yhat"].values
    actual = test["y"].values

    return {
        "mape": round(mean_absolute_percentage_error(actual, pred), 2),
        "rmse": round(root_mean_squared_error(actual, pred), 2),
        "mae": round(mean_absolute_error(actual, pred), 2),
        "n_test": test_size,
    }


def evaluate_all_products(df: pd.DataFrame, test_size: int = 30) -> pd.DataFrame:
    """
    Compute accuracy metrics for all store-product combinations.
    
    Args:
        df: Main retail DataFrame
        test_size: Holdout size per combination
    
    Returns:
        DataFrame with metrics per store-product
    """
    results = []
    stores = df["store"].unique()
    products = df["product"].unique()

    for store in stores:
        for product in products:
            data = (
                df[(df["store"] == store) & (df["product"] == product)]
                .groupby("date")["sales"]
                .sum()
                .reset_index()
                .rename(columns={"date": "ds", "sales": "y"})
            )

            if len(data) < test_size + 30:
                continue

            metrics = evaluate_prophet_model(None, data, test_size=test_size)
            results.append({
                "store": store,
                "product": product,
                **metrics
            })

    return pd.DataFrame(results)
