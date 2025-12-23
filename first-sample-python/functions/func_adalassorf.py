"""
Adaptive LASSO + Random Forest hybrid function for inflation forecasting.
Converted from first-sample/functions/func-adalassorf.R with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed
from functions.func_lasso import ic_glmnet_bic


def runlasso(Y, indice, lag, alpha=1, type="lasso"):
    """
    Run LASSO for variable selection, then apply Random Forest.
    This is the func-adalassorf version.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
    alpha : float
        Elastic net mixing parameter
    type : str
        Type of LASSO: "lasso", "adalasso", or "fal"
        
    Returns:
    --------
    dict with keys:
        - pred: prediction from Random Forest on selected variables
    """
    # Center the data (scale=FALSE means center only)
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
    
    # Fit initial LASSO model for variable selection
    result = ic_glmnet_bic(X, y, alpha=alpha)
    coef = result["coef"]
    
    if type == "adalasso":
        # Adaptive Lasso
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        result = ic_glmnet_bic(X, y, penalty_factor=penalty, alpha=alpha)
        coef = result["coef"]
    
    elif type == "fal":
        # Flexible Adaptive Lasso
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = np.arange(0, 1.1, 0.1)
        bb = np.inf
        best_coef = coef
        
        for alpha_val in alphas:
            m0 = ic_glmnet_bic(X, y, alpha=alpha_val)
            coef_temp = m0["coef"]
            
            for tau in taus:
                penalty = (np.abs(coef_temp[1:]) + 1/np.sqrt(len(y))) ** (-tau)
                m = ic_glmnet_bic(X, y, penalty_factor=penalty, alpha=1.0)
                crit = m["bic"]
                
                if crit < bb:
                    best_coef = m["coef"]
                    bb = crit
        
        coef = best_coef
    
    # Select variables with non-zero coefficients
    selected = np.where(coef[1:] != 0)[0]
    
    # Ensure at least 2 variables are selected
    if len(selected) < 2:
        selected = np.array([0, 1])
    
    # Fit Random Forest on selected variables
    modelrf = RandomForestRegressor(n_estimators=500, random_state=42)
    modelrf.fit(X[:, selected], y)
    
    # Predict
    pred = modelrf.predict(X_out[selected].reshape(1, -1))[0]
    
    return {"pred": pred}


def lasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1, type="lasso"):
    """
    Rolling window Adaptive LASSO + Random Forest forecasting.
    
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
    alpha : float
        Elastic net mixing parameter
    type : str
        Type of model: "lasso", "adalasso", or "fal"
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - errors: RMSE and MAE
    """
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        lasso = runlasso(Y_window, indice, lag, alpha, type)
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
    plt.title(f'Adaptive LASSO + Random Forest Forecasting - {type}')
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
