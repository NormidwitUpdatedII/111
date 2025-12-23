"""
Bagging functions for inflation forecasting.
Converted from R to Python with exact correspondence.
Note: Implements bagging with pre-testing similar to HDeconometrics::bagging
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def bagging_pretesting(X, y, R=100, l=5):
    """
    Bagging with pre-testing.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix
    y : numpy.ndarray
        Response variable
    R : int
        Number of bootstrap samples
    l : int
        Block size for pre-testing
        
    Returns:
    --------
    dict with coefficients and model info
    """
    n, p = X.shape
    all_coefs = np.zeros((R, p + 1))  # +1 for intercept
    
    for r in range(R):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, replace=True, random_state=r)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        
        all_coefs[r, 0] = model.intercept_
        all_coefs[r, 1:] = model.coef_
    
    # Average coefficients
    avg_coef = np.mean(all_coefs, axis=0)
    
    return {"coef_matrix": all_coefs, "avg_coef": avg_coef}


def runbagg(Y, indice, lag):
    """
    Run bagging regression.
    
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
        - model: bagging model info
        - pred: prediction
        - nselect: variable selection counts
    """
    Y_centered = Y - np.mean(Y, axis=0)
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    Y2 = np.column_stack([Y, scores[:, :4]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]
    
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = bagging_pretesting(X, y, R=100, l=5)
    
    # Prediction using average coefficients
    pred = np.dot(np.concatenate([[1], X_out]), model["avg_coef"])
    
    # Count variable selections
    nselect = np.sum(model["coef_matrix"][:, 1:] != 0, axis=0)
    
    return {"model": model, "pred": pred, "nselect": nselect}


def bagg_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window bagging forecasting.
    
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
        - errors: RMSE and MAE
        - nselect: variable selection counts
    """
    n_features = 21 + (Y.shape[1] - 1) * 4
    save_nselect = np.full((nprev, n_features), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        bagg = runbagg(Y_window, indice, lag)
        save_pred[nprev - i, 0] = bagg["pred"]
        save_nselect[nprev - i, :min(len(bagg["nselect"]), n_features)] = \
            bagg["nselect"][:n_features]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'Bagging Forecasting (lag={lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "errors": errors, "nselect": save_nselect}
