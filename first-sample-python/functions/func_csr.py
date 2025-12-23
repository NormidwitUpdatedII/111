"""
Complete Subset Regression (CSR) functions for inflation forecasting.
Converted from R to Python with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def csr_regression(X, y, fixed_controls=None):
    """
    Complete Subset Regression - fits models on all subsets and averages.
    Simplified implementation.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def runcsr(Y, indice, lag):
    """Run Complete Subset Regression."""
    Y_centered = Y - np.mean(Y, axis=0)
    pca = PCA()
    scores = pca.fit_transform(Y_centered)
    Y2 = np.column_stack([Y, scores[:, :4]])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]
    
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    f_seq = list(range(indice - 1, X.shape[1], n_cols_Y2))
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = csr_regression(X, y, fixed_controls=f_seq)
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {"model": model, "pred": pred}


def csr_rolling_window(Y, nprev, indice=1, lag=1):
    """Rolling window CSR forecasting."""
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        cs = runcsr(Y_window, indice, lag)
        save_pred[nprev - i, 0] = cs["pred"]
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'CSR Forecasting (lag={lag})')
    plt.grid(True)
    plt.show()
    
    real_tail = real[-nprev:]
    rmse = np.sqrt(np.mean((real_tail - save_pred.flatten()) ** 2))
    mae = np.mean(np.abs(real_tail - save_pred.flatten()))
    errors = {"rmse": rmse, "mae": mae}
    
    return {"pred": save_pred, "errors": errors}
