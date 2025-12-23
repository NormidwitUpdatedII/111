"""
Random Forest Factor (RFF) function for inflation forecasting.
This is a 2-stage method:
1. Uses importance scores from a prior Random Forest run
2. Selects top features based on importance
3. Runs factor model on selected features

Note: The original R func-rffact.R is missing from the repository.
This implementation is reverse-engineered from how it's called in rff.R:
    rffact.rolling.window(Y, nprev, indice, lag, rf_importance)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def runrffact(Y, indice, lag, importance, n_select=50):
    """
    Run RF-Factor model using pre-computed importance scores.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed to match R)
    lag : int
        Number of lags
    importance : numpy.ndarray
        Feature importance scores from Random Forest
    n_select : int
        Number of top features to select (default 50)
        
    Returns:
    --------
    dict with keys:
        - pred: prediction
        - coef: coefficients
    """
    # Center the data
    Y_centered = Y - np.mean(Y, axis=0)
    
    # PCA on full data
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
    
    # Use importance to select top features
    # importance can be a list (one per rolling window iteration) or single array
    if isinstance(importance, list):
        # Average importance across iterations
        avg_importance = np.mean([imp if isinstance(imp, np.ndarray) else np.array(imp) 
                                   for imp in importance], axis=0)
    else:
        avg_importance = importance
    
    # Select top n_select features
    n_select = min(n_select, len(avg_importance), X.shape[1])
    top_indices = np.argsort(avg_importance)[::-1][:n_select]
    
    # Select features
    X_selected = X[:, top_indices]
    X_out_selected = X_out[top_indices]
    
    # Run factor model on selected features
    # Use BIC to select number of factors (similar to func-fact.R)
    bb = np.inf
    best_model = None
    best_coef = None
    
    for n_factors in range(5, min(21, X_selected.shape[1] + 1), 5):
        # PCA for factor extraction
        pca_fact = PCA(n_components=n_factors)
        factors = pca_fact.fit_transform(X_selected)
        
        # Fit regression
        model = LinearRegression()
        model.fit(factors, y)
        
        # Calculate BIC
        y_pred = model.predict(factors)
        rss = np.sum((y - y_pred) ** 2)
        n = len(y)
        k = n_factors + 1
        bic = n * np.log(rss / n) + k * np.log(n)
        
        if bic < bb:
            bb = bic
            best_model = model
            best_pca = pca_fact
            best_coef = np.concatenate([[model.intercept_], model.coef_])
    
    # Prediction
    factors_out = best_pca.transform(X_out_selected.reshape(1, -1))
    pred = best_model.predict(factors_out)[0]
    
    # Pad coefficients to standard size
    coef = np.zeros(21)
    coef[:len(best_coef)] = best_coef
    
    return {"pred": pred, "coef": coef}


def rffact_rolling_window(Y, nprev, indice=1, lag=1, importance_list=None):
    """
    Rolling window RF-Factor forecasting.
    
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
    importance_list : list
        List of importance arrays from RF rolling window
        (one per rolling window iteration)
        
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
        
        # Get importance for this iteration
        if importance_list is not None:
            # importance_list is indexed from the RF run
            importance = importance_list[nprev - i] if (nprev - i) < len(importance_list) else importance_list[-1]
        else:
            # If no importance provided, use uniform weights
            importance = np.ones(Y.shape[1] + 4)  # +4 for PC scores
        
        result = runrffact(Y_window, indice, lag, importance)
        save_coef[nprev - i, :] = result["coef"]
        save_pred[nprev - i, 0] = result["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'RF-Factor Forecasting (lag={lag})')
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
