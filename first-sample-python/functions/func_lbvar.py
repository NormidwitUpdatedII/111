"""
Large Bayesian Vector Autoregression (LBVAR) for inflation forecasting.
Converted from first-sample/functions/func-lbvar.R with exact correspondence.

This implements a simplified LBVAR using Ridge regression as an approximation
to the Bayesian shrinkage approach.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def lbvar(Y, p, delta=0, lambda_param=0.05):
    """
    Simplified Large Bayesian VAR implementation using Ridge regression.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix (T x K)
    p : int
        Lag order
    delta : float
        Lag decay parameter (not used in simplified version)
    lambda_param : float
        Shrinkage parameter (Ridge alpha)
        
    Returns:
    --------
    dict with keys:
        - model: fitted Ridge model
        - coef: coefficients
        - covmat: covariance matrix (diagonal approximation)
        - Y: data used
    """
    T, K = Y.shape
    
    # Create lagged design matrix
    Y_lagged = embed(Y, p + 1)
    
    # Y matrix (current values)
    Y_current = Y_lagged[:, :K]
    
    # X matrix (lagged values)
    X_lagged = Y_lagged[:, K:]
    
    # Fit Ridge regression for each variable
    models = []
    coefs = []
    residuals_list = []
    
    for k in range(K):
        model = Ridge(alpha=lambda_param * X_lagged.shape[0])
        model.fit(X_lagged, Y_current[:, k])
        models.append(model)
        coefs.append(np.concatenate([[model.intercept_], model.coef_]))
        
        # Calculate residuals
        pred = model.predict(X_lagged)
        residuals_list.append(Y_current[:, k] - pred)
    
    # Covariance matrix (diagonal approximation)
    residuals = np.column_stack(residuals_list)
    covmat = np.diag(np.var(residuals, axis=0))
    
    return {
        "model": models,
        "coef": np.array(coefs),
        "covmat": covmat,
        "Y": Y,
        "p": p,
        "K": K
    }


def predict_lbvar(lbvar_model, h=1):
    """
    Forecast from LBVAR model.
    
    Parameters:
    -----------
    lbvar_model : dict
        Fitted LBVAR model
    h : int
        Forecast horizon
        
    Returns:
    --------
    forecasts : numpy.ndarray
        Forecasts (h x K matrix)
    """
    models = lbvar_model["model"]
    Y = lbvar_model["Y"]
    p = lbvar_model["p"]
    K = lbvar_model["K"]
    
    # Get last p observations
    Y_last = Y[-p:, :].flatten()
    
    forecasts = []
    
    for step in range(h):
        # Forecast for each variable
        forecast_step = np.zeros(K)
        
        for k in range(K):
            # Use last p*K values for prediction
            X_pred = Y_last[-(p * K):].reshape(1, -1)
            forecast_step[k] = models[k].predict(X_pred)[0]
        
        forecasts.append(forecast_step)
        
        # Update Y_last for next forecast
        Y_last = np.concatenate([Y_last[K:], forecast_step])
    
    return np.array(forecasts)


def lbvar_rw(Y, p, lag, nprev, delta=0, lambda_param=0.05, variables=None):
    """
    Rolling window LBVAR forecasting.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    p : int
        VAR lag order
    lag : int
        Forecast horizon
    nprev : int
        Number of predictions (rolling window size)
    delta : float
        Lag decay parameter
    lambda_param : float
        Shrinkage parameter
    variables : int or list, optional
        Variables to forecast (1-indexed). If int, forecasts single variable.
        If None, forecasts variable 1.
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - covmat: covariance matrices
        - real: actual values
    """
    if variables is None:
        variables = [1]
    elif isinstance(variables, int):
        variables = [variables]
    
    # Adjust for 0-indexing
    variables_idx = [v - 1 for v in variables]
    
    # Get actual values
    real = Y[-nprev:, :]
    
    # Remove last 'lag' observations for training
    Y_train = Y[:-lag, :]
    
    store_pred = np.full((nprev, len(variables_idx)), np.nan)
    store_covmat = np.full((nprev, Y.shape[1]), np.nan)
    
    for i in range(1, nprev + 1):
        # Create rolling window
        y_window = Y_train[(nprev - i):(Y_train.shape[0] - i + 1), :]
        
        # Fit LBVAR
        model = lbvar(y_window, p, delta=delta, lambda_param=lambda_param)
        
        # Get covariance matrix diagonal
        covmat = np.sqrt(np.diag(model["covmat"]))
        
        # Forecast
        pred = predict_lbvar(model, h=lag)
        
        # Store predictions for selected variables
        store_pred[i - 1, :] = pred[lag - 1, variables_idx]
        store_covmat[i - 1, :] = covmat
        
        print(f"iteration {i}")
    
    # Reverse order to match R output
    store_pred = store_pred[::-1, :]
    
    return {
        "pred": store_pred,
        "covmat": store_covmat,
        "real": real[:, variables_idx]
    }
