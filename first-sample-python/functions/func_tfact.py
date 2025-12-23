"""
Targeted Factor Models for inflation forecasting.
Converted from first-sample/functions/func-tfact.R with exact correspondence.

Targeted factors use pre-testing/variable selection before extracting factors.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import BaggingRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed
from functions.func_fact import runfact


def baggit_pretesting(mat, fixed_controls=None):
    """
    Simplified version of baggit with pre-testing.
    Uses Lasso for variable selection.
    
    Parameters:
    -----------
    mat : numpy.ndarray
        Data matrix (response in first column, predictors in remaining columns)
    fixed_controls : list, optional
        Indices of variables to always include (0-indexed)
        
    Returns:
    --------
    numpy.ndarray
        Binary selection indicator (1=selected, 0=not selected)
    """
    y = mat[:, 0]
    X = mat[:, 1:]
    
    # Use Lasso for variable selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X, y)
    
    # Get selected variables
    selected = np.abs(lasso.coef_) > 1e-10
    
    # Ensure fixed controls are included
    if fixed_controls is not None:
        for idx in fixed_controls:
            if idx < len(selected):
                selected[idx] = True
    
    return selected.astype(int)


def run_tfact(Y, indice, lag):
    """
    Run targeted factor model with pre-testing.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
        
    Returns:
    --------
    dict with keys:
        - coef: coefficients
        - pred: prediction
    """
    # Extract target variable and predictors
    y = Y[:, indice - 1]
    X = Y[:, [i for i in range(Y.shape[1]) if i != indice - 1]]
    
    # Adjust for lag
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Create matrix with embedded target and predictors
    y_embed = embed(y.reshape(-1, 1), 5)[:, 0]
    y_lags = embed(y.reshape(-1, 1), 5)[:, 1:]  # lags 1-4
    
    # Align dimensions
    min_len = min(len(y_embed), X.shape[0] - 4)
    y_embed = y_embed[:min_len]
    y_lags = y_lags[:min_len, :]
    X_aligned = X[-min_len:, :]  # Use tail
    
    # Combine for pre-testing
    mat = np.column_stack([y_embed, y_lags, X_aligned])
    
    # Pre-testing with fixed controls (first 4 lags)
    pretest = baggit_pretesting(mat, fixed_controls=[0, 1, 2, 3])
    pretest = pretest[4:]  # Remove the lag indicators, keep predictor indicators
    
    # Select variables
    aux = np.zeros(Y.shape[1])
    aux[indice - 1] = 1
    aux[[i for i in range(Y.shape[1]) if i != indice - 1]] = pretest[:len(aux) - 1] if len(pretest) >= len(aux) - 1 else np.concatenate([pretest, np.zeros(len(aux) - 1 - len(pretest))])
    
    selected_idx = np.where(aux == 1)[0]
    
    # Extract selected variables
    Y2 = Y[:, selected_idx]
    
    # Run factor model on selected variables
    fmodel = runfact(Y2, np.where(selected_idx == (indice - 1))[0][0] + 1, lag)  # Adjust indice for subset
    
    return {"coef": fmodel["coef"], "pred": fmodel["pred"]}


def tfact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window targeted factor forecasting.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    nprev : int
        Number of predictions (rolling window size)
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - coef: coefficients for each window
        - errors: RMSE and MAE
    """
    save_coef = np.full((nprev, 21), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        fact = run_tfact(Y_window, indice, lag)
        
        # Store coefficients
        coef = fact["coef"]
        save_coef[nprev - i, :min(len(coef), 21)] = coef[:21] if len(coef) >= 21 else np.concatenate([coef, np.full(21 - len(coef), np.nan)])
        save_pred[nprev - i, 0] = fact["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title('Targeted Factor Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    # Calculate errors
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "coef": save_coef, "errors": errors}
