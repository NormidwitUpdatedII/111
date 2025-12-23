"""
LASSO and related regularized regression functions for inflation forecasting.
Converted from R to Python with exact correspondence.
Note: ic.glmnet in R is replaced with cross-validation in scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def ic_glmnet_bic(X, y, alpha=1.0, penalty_factor=None):
    """
    Fit Lasso/ElasticNet using cross-validation and BIC-like criterion.
    Approximates R's ic.glmnet function.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix
    y : numpy.ndarray
        Response variable
    alpha : float
        Elastic net mixing parameter (1=Lasso, 0=Ridge)
    penalty_factor : numpy.ndarray, optional
        Penalty factors for each feature
        
    Returns:
    --------
    dict with keys:
        - model: fitted model
        - coef: coefficients (including intercept)
        - bic: BIC value
    """
    if alpha == 1.0:
        # Pure Lasso
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
    else:
        # Elastic Net
        model = ElasticNetCV(l1_ratio=alpha, cv=5, random_state=42, max_iter=10000)
    
    # Apply penalty factors if provided (approximate by scaling features)
    if penalty_factor is not None:
        X_scaled = X * penalty_factor.reshape(1, -1)
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)
    
    # Get coefficients
    coef = np.concatenate([[model.intercept_], model.coef_])
    
    # Calculate BIC
    y_pred = model.predict(X if penalty_factor is None else X_scaled)
    rss = np.sum((y - y_pred) ** 2)
    n = len(y)
    k = np.sum(model.coef_ != 0) + 1  # number of non-zero coefficients + intercept
    bic = n * np.log(rss / n) + k * np.log(n)
    
    return {"model": model, "coef": coef, "bic": bic}


def runlasso(Y, indice, lag, alpha=1, type="lasso"):
    """
    Run LASSO regression with optional adaptive penalties.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed to match R)
    lag : int
        Number of lags
    alpha : float
        Elastic net mixing parameter (1=Lasso, 0=Ridge)
    type : str
        Type of model: "lasso", "adalasso", or "fal"
        
    Returns:
    --------
    dict with keys:
        - model: fitted model or dict with model info
        - pred: prediction
    """
    # Center the data (scale=FALSE in R means center only)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # PCA
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    
    # Combine original data with first 4 principal components
    Y2 = np.column_stack([Y, scores[:, :4]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]  # Adjust for 0-indexing
    
    # Remove first (ncol(Y2)*lag) columns
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Fit initial model
    result = ic_glmnet_bic(X, y, alpha=alpha)
    model_info = result
    coef = result["coef"]
    
    if type == "adalasso":
        # Adaptive Lasso
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        result = ic_glmnet_bic(X, y, penalty_factor=penalty, alpha=alpha)
        model_info = result
    
    elif type == "fal":
        # Flexible Adaptive Lasso with grid search
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = np.arange(0, 1.1, 0.1)
        bb = np.inf
        best_model = None
        
        for alpha_val in alphas:
            m0 = ic_glmnet_bic(X, y, alpha=alpha_val)
            coef_temp = m0["coef"]
            
            for tau in taus:
                penalty = (np.abs(coef_temp[1:]) + 1/np.sqrt(len(y))) ** (-tau)
                m = ic_glmnet_bic(X, y, penalty_factor=penalty, alpha=1.0)
                crit = m["bic"]
                
                if crit < bb:
                    best_model = m
                    bb = crit
        
        model_info = best_model
    
    # Prediction
    model = model_info["model"]
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {"model": model_info, "pred": pred}


def lasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1, type="lasso"):
    """
    Rolling window LASSO forecasting.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    nprev : int
        Number of predictions (rolling window size)
    indice : int
        Column index to forecast (1-indexed to match R)
    lag : int
        Number of lags
    alpha : float
        Elastic net mixing parameter
    type : str
        Type of model: "lasso", "adalasso", or "fal"
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - coef: coefficients for each window
        - errors: RMSE and MAE
    """
    n_features = 21 + (Y.shape[1] - 1) * 4
    save_coef = np.full((nprev, n_features), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        lasso = runlasso(Y_window, indice, lag, alpha, type)
        
        # Store coefficients
        coef = lasso["model"]["coef"]
        save_coef[nprev - i, :min(len(coef), n_features)] = coef[:n_features]
        save_pred[nprev - i, 0] = lasso["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'LASSO (alpha={alpha}) Forecasting - {type}')
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


def runpols(Y, indice, lag, coef):
    """
    Run post-LASSO OLS (re-fit selected variables with OLS).
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
    coef : numpy.ndarray
        Coefficients from LASSO (to identify selected variables)
        
    Returns:
    --------
    dict with keys:
        - model: OLS model
        - pred: prediction
    """
    # Center the data
    Y_centered = Y - np.mean(Y, axis=0)
    
    # PCA
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
    
    # Select variables with non-zero coefficients
    selected = np.where(coef[1:] != 0)[0]
    
    if len(selected) == 0:
        # Intercept-only model
        model = LinearRegression()
        model.fit(np.ones((len(y), 1)), y)
        pred = model.intercept_
    else:
        respo = X[:, selected]
        model = LinearRegression()
        model.fit(respo, y)
        pred = np.dot(np.concatenate([[1], X_out[selected]]), 
                     np.concatenate([[model.intercept_], model.coef_]))
    
    return {"model": model, "pred": pred}


def pols_rolling_window(Y, nprev, indice=1, lag=1, coef=None):
    """
    Rolling window post-LASSO OLS forecasting.
    
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
    coef : numpy.ndarray
        Matrix of LASSO coefficients (one row per window)
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - errors: RMSE and MAE
    """
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        m = runpols(Y_window, indice, lag, coef[nprev - i, :])
        save_pred[nprev - i, 0] = m["pred"]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title('Post-LASSO OLS Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    # Calculate errors
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "errors": errors}
