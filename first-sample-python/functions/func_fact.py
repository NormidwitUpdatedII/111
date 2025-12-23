"""
Factor model functions for inflation forecasting.
Converted from R to Python with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed, calculate_bic


def runfact(Y, indice, lag):
    """
    Run factor model with BIC selection.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed to match R)
    lag : int
        Number of lags
        
    Returns:
    --------
    dict with keys:
        - model: fitted model
        - pred: prediction
        - coef: coefficients
    """
    Y_centered = Y - np.mean(Y, axis=0)
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    
    # Use target variable and first 4 principal components
    Y2 = np.column_stack([Y[:, indice - 1], scores[:, :4]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, 0]
    
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # BIC selection over factor counts: 5, 10, 15, 20
    bb = np.inf
    best_model = None
    f_coef = None
    
    for i in range(5, 21, 5):
        if i > X.shape[1]:
            i = X.shape[1]
        
        m = LinearRegression()
        m.fit(X[:, :i], y)
        crit = calculate_bic(y, X[:, :i], m)
        
        if crit < bb:
            bb = crit
            best_model = m
            f_coef = np.concatenate([[m.intercept_], m.coef_])
    
    model = best_model
    coef = np.zeros(X.shape[1] + 1)
    coef[:len(f_coef)] = f_coef
    
    # Prediction
    pred = np.dot(np.concatenate([[1], X_out]), coef)
    
    return {"model": model, "pred": pred, "coef": coef}


def fact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window factor model forecasting.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    nprev : int
        Number of predictions
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
        fact = runfact(Y_window, indice, lag)
        save_coef[nprev - i, :] = fact["coef"][:21]
        save_pred[nprev - i, 0] = fact["pred"]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'Factor Model Forecasting (lag={lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "coef": save_coef, "errors": errors}
