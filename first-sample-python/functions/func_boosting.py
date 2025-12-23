"""
Boosting functions for inflation forecasting.
Converted from R to Python with exact correspondence.
Implements gradient boosting similar to HDeconometrics::boosting
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def boosting(y, X, v=0.2, M=100):
    """
    Gradient boosting implementation.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Response variable
    X : numpy.ndarray
        Design matrix
    v : float
        Learning rate (shrinkage parameter)
    M : int
        Number of boosting iterations
        
    Returns:
    --------
    dict with coefficients and model info
    """
    n, p = X.shape
    
    # Initialize
    f = np.mean(y) * np.ones(n)
    coef = np.zeros(p + 1)
    coef[0] = np.mean(y)
    
    for m in range(M):
        # Calculate residuals
        r = y - f
        
        # Fit weak learner to residuals
        model = LinearRegression()
        model.fit(X, r)
        
        # Update function
        f = f + v * model.predict(X)
        
        # Update coefficients
        coef[0] += v * model.intercept_
        coef[1:] += v * model.coef_
    
    return {"coef": coef, "f": f}


def runboost(Y, indice, lag):
    """
    Run boosting regression.
    
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
        - model: boosting model
        - pred: prediction
    """
    Y_centered = Y - np.mean(Y, axis=0)
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    
    # Use target variable and first 8 principal components
    Y2 = np.column_stack([Y[:, indice - 1], scores[:, :8]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, 0]  # First column is the target
    
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = boosting(y, X, v=0.2)
    
    # Prediction
    pred = np.dot(np.concatenate([[1], X_out]), model["coef"])
    
    return {"model": model, "pred": pred}


def boosting_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window boosting forecasting.
    
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
    save_coef = np.full((nprev, 36), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        bo = runboost(Y_window, indice, lag)
        save_coef[nprev - i, :min(len(bo["model"]["coef"]), 36)] = \
            bo["model"]["coef"][:36]
        save_pred[nprev - i, 0] = bo["pred"]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'Boosting Forecasting (lag={lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "coef": save_coef, "errors": errors}
