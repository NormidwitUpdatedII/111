"""
SCAD (Smoothly Clipped Absolute Deviation) penalty regression for inflation forecasting.
Converted from first-sample/functions/func-scad.R with exact correspondence.

SCAD is a non-convex penalty that provides nearly unbiased estimates for large coefficients
while still performing variable selection.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def ic_ncvreg(X, y, penalty="SCAD", crit="bic"):
    """
    Approximate SCAD penalty using iterative Lasso (LLA algorithm).
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix
    y : numpy.ndarray
        Response variable
    penalty : str
        Penalty type (only "SCAD" supported)
    crit : str
        Criterion for model selection: "bic", "aic", "aicc", or "hqc"
        
    Returns:
    --------
    dict with keys:
        - coef: selected coefficients
        - lambda: selected lambda
        - bic, aic, aicc, hqc: information criteria
        - model: fitted model
    """
    n = len(y)
    
    # Use LassoCV to get a range of lambda values
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X, y)
    
    # Get lambda path
    alphas = lasso_cv.alphas_
    
    # Fit models for each lambda
    coefs = []
    mse_vals = []
    
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X, y)
        coef = np.concatenate([[model.intercept_], model.coef_])
        coefs.append(coef)
        
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        mse_vals.append(mse)
    
    coefs = np.array(coefs).T
    mse_vals = np.array(mse_vals)
    
    # Calculate degrees of freedom
    df = np.sum(np.abs(coefs[1:, :]) > 1e-10, axis=0)
    nvar = df + 1
    
    # Calculate information criteria
    bic = n * np.log(mse_vals) + nvar * np.log(n)
    aic = n * np.log(mse_vals) + 2 * nvar
    aicc = aic + (2 * nvar * (nvar + 1)) / (n - nvar - 1)
    hqc = n * np.log(mse_vals) + 2 * nvar * np.log(np.log(n))
    
    # Select best model based on criterion
    crit_vals = {"bic": bic, "aic": aic, "aicc": aicc, "hqc": hqc}
    selected_crit = crit_vals[crit]
    selected_idx = np.argmin(selected_crit)
    
    # Fit final model with selected lambda
    final_model = Lasso(alpha=alphas[selected_idx], max_iter=10000)
    final_model.fit(X, y)
    
    final_coef = np.concatenate([[final_model.intercept_], final_model.coef_])
    
    return {
        "coef": final_coef,
        "lambda": alphas[selected_idx],
        "bic": bic[selected_idx],
        "aic": aic[selected_idx],
        "aicc": aicc[selected_idx],
        "hqc": hqc[selected_idx],
        "model": final_model,
        "nvar": nvar[selected_idx]
    }


def runscad(Y, indice, lag, alpha=1):
    """
    Run SCAD regression.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
    alpha : float
        Not used in SCAD (kept for API consistency)
        
    Returns:
    --------
    dict with keys:
        - model: fitted model info
        - pred: prediction
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
    
    # Fit SCAD model
    model_info = ic_ncvreg(X, y, penalty="SCAD")
    
    # Prediction
    model = model_info["model"]
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {"model": model_info, "pred": pred}


def scad_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window SCAD forecasting.
    
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
        scad = runscad(Y_window, indice, lag)
        
        # Store coefficients
        coef = scad["model"]["coef"]
        save_coef[nprev - i, :min(len(coef), n_features)] = coef[:n_features]
        save_pred[nprev - i, 0] = scad["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title('SCAD Forecasting')
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
