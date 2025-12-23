"""
XGBoost functions for inflation forecasting.
Converted from R to Python with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import xgboost as xgb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def runxgb(Y, indice, lag):
    """
    Run XGBoost regression.
    
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
        - model: fitted XGBoost model
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
    
    # XGBoost parameters matching R implementation
    # params: eta=0.05, nthread=1, colsample_bylevel=2/3, subsample=1, 
    #         max_depth=4, min_child_weight=nrow(X)/200
    params = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 1.0,
        'colsample_bylevel': 2/3,
        'min_child_weight': X.shape[0] / 200,
        'objective': 'reg:squarederror',
        'nthread': 1,
        'seed': 42
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        verbose_eval=False
    )
    
    # Prediction (note: R uses t(X.out) which makes it a column vector)
    dtest = xgb.DMatrix(X_out.reshape(1, -1))
    pred = model.predict(dtest)[0]
    
    return {"model": model, "pred": pred}


def xgb_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window XGBoost forecasting.
    
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
    """
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        xgb_result = runxgb(Y_window, indice, lag)
        save_pred[nprev - i, 0] = xgb_result["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'XGBoost Forecasting (lag={lag})')
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
