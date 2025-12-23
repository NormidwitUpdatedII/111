"""
Polynomial LASSO (LASSO with polynomial interactions) for inflation forecasting.
Converted from first-sample/functions/func-polilasso.R with exact correspondence.

This creates polynomial interactions of features before applying LASSO.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed
from functions.func_lasso import ic_glmnet_bic


def runlasso(Y, indice, lag, alpha=1, type="lasso"):
    """
    Run LASSO regression with polynomial interactions.
    Creates pairwise products of features within each lag block.
    
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
        Type of model: "lasso" or "adalasso"
        
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
    
    # Create polynomial interactions
    # In R code: z = ncol(X)/4, creates pairwise products within each quarter
    z = X.shape[1] // 4
    
    comb1_list = []
    comb1out_list = []
    
    # For each of the 4 quarters of features
    for u in range(4):
        start_idx = u * 126
        end_idx = (u + 1) * 126
        
        # Handle case where X might not have exactly 504 columns (4*126)
        if end_idx > X.shape[1]:
            end_idx = X.shape[1]
        if start_idx >= X.shape[1]:
            break
            
        comb0 = X[:, start_idx:end_idx]
        comb0out = X_out[start_idx:end_idx]
        
        # Create all pairwise products
        for i in range(comb0.shape[1]):
            for j in range(comb0.shape[1]):
                comb1_list.append(comb0[:, i] * comb0[:, j])
                comb1out_list.append(comb0out[i] * comb0out[j])
    
    # Combine all polynomial features
    comb1 = np.column_stack(comb1_list) if comb1_list else X
    comb1out = np.array(comb1out_list) if comb1out_list else X_out
    
    # Fit LASSO model
    result = ic_glmnet_bic(comb1, y, alpha=alpha)
    model_info = result
    coef = result["coef"]
    
    if type == "adalasso":
        # Adaptive Lasso
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        result = ic_glmnet_bic(comb1, y, penalty_factor=penalty, alpha=alpha)
        model_info = result
    
    # Prediction
    model = model_info["model"]
    pred = model.predict(comb1out.reshape(1, -1))[0]
    
    return {"model": model_info, "pred": pred}


def lasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1, type="lasso"):
    """
    Rolling window Polynomial LASSO forecasting.
    
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
        Type of model: "lasso" or "adalasso"
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - coef: coefficients for each window
        - errors: RMSE and MAE
    """
    # Estimate coefficient matrix size
    n_base_features = 20 + (Y.shape[1] - 1) * 4
    n_poly_features = n_base_features * 126  # From R code structure
    
    save_coef = np.full((nprev, 1 + n_poly_features), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        lasso = runlasso(Y_window, indice, lag, alpha, type)
        
        # Store coefficients
        coef = lasso["model"]["coef"]
        save_coef[nprev - i, :min(len(coef), save_coef.shape[1])] = coef[:save_coef.shape[1]]
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
    plt.title(f'Polynomial LASSO Forecasting - {type}')
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
