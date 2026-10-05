import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, and R2 score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    residuals = (y_true - y_pred).tolist()

    return {
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "residuals": [round(float(r), 2) for r in residuals[:150]] # sample for frontend visualization
    }
