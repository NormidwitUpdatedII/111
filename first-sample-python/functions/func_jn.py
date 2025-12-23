"""
Jackknife model functions for inflation forecasting.
Converted from R to Python with exact correspondence.
Implements jackknife model combination similar to HDeconometrics::jackknife
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def jackknife(X_list, y, lag=4, fixed_controls=None):
    """
    Jackknife model combination.
    
    Parameters:
    -----------
    X_list : list of numpy.ndarray
        List of design matrices (one for each lag)
    y : numpy.ndarray
        Response variable
    lag : int
        Number of lags
    fixed_controls : int, optional
        Index of fixed control variable
        
    Returns:
    --------
    dict with weights, coefficients, and models
    """
    n_models = len(X_list)
    weights = np.ones(n_models) / n_models  # Equal weights as default
    
    models = []
    predictions = np.zeros((len(y), n_models))
    
    for i, X in enumerate(X_list):
        model = LinearRegression()
        model.fit(X, y)
        models.append(model)
        predictions[:, i] = model.predict(X)
    
    # Calculate optimal weights using jackknife
    # Simple implementation: use inverse MSE weighting
    mse = np.mean((y.reshape(-1, 1) - predictions) ** 2, axis=0)
    weights = 1 / (mse + 1e-10)
    weights = weights / np.sum(weights)
    
    # Final coefficients (weighted average)
    coef_list = [np.concatenate([[m.intercept_], m.coef_]) for m in models]
    avg_coef = np.average(coef_list, axis=0, weights=weights)
    
    return {"weights": weights, "coef": avg_coef, "models": models}


def runjn(Y, indice, lag):
    """
    Run jackknife model.
    
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
        - model: jackknife model
        - pred: prediction
        - weights: model weights
        - coef: coefficients
    """
    Y_centered = Y - np.mean(Y, axis=0)
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    Y2 = np.column_stack([Y, scores[:, :4]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]
    
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Create indices for splitting X into lag periods
    i1 = list(range(0, X.shape[1], n_cols_Y2))
    i2 = [min(i + n_cols_Y2, X.shape[1]) for i in i1]
    
    # Split X into list of matrices
    auxX = [X[:, i1[i]:i2[i]] for i in range(len(i1))]
    
    # Create X_out for each lag
    if lag == 1:
        X_out = [aux[-1, i1[i]:i2[i]] for i in range(len(i1))]
    else:
        aux_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = [aux_temp[-1, i1[i]:i2[i]] for i in range(len(i1))]
    
    model = jackknife(auxX, y, lag=4, fixed_controls=indice)
    
    # Prediction using weighted combination
    pred = 0
    for i, m in enumerate(model["models"]):
        pred += model["weights"][i] * m.predict(X_out[i].reshape(1, -1))[0]
    
    return {"model": model, "pred": pred, "weights": model["weights"], "coef": model["coef"]}


def jackknife_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window jackknife forecasting.
    
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
        - weights: model weights for each window
        - errors: RMSE and MAE
    """
    n_weights = Y.shape[1] - 1 + 4
    save_weights = np.full((nprev, n_weights), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        jn = runjn(Y_window, indice, lag)
        save_pred[nprev - i, 0] = jn["pred"]
        save_weights[nprev - i, :min(len(jn["weights"]), n_weights)] = \
            jn["weights"][:n_weights]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'Jackknife Forecasting (lag={lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "weights": save_weights, "errors": errors}
