"""
Random Forest functions for inflation forecasting.
Converted from R to Python with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def runrf(Y, indice, lag):
    """
    Run Random Forest regression.
    
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
        - model: fitted Random Forest model
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
    
    # Fit Random Forest with importance calculation
    # Default parameters similar to R's randomForest
    model = RandomForestRegressor(
        n_estimators=500,  # default in R
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Prediction
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {"model": model, "pred": pred}


def rf_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Random Forest forecasting.
    
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
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - errors: RMSE and MAE
        - save_importance: list of feature importances for each window
    """
    save_importance = []
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        rf_result = runrf(Y_window, indice, lag)
        save_pred[nprev - i, 0] = rf_result["pred"]
        save_importance.append(rf_result["model"].feature_importances_)
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'Random Forest Forecasting (lag={lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    # Calculate errors
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "errors": errors, "save_importance": save_importance}
