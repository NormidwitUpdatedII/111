"""
Autoregressive (AR) model functions for inflation forecasting.
Converted from R to Python with exact correspondence.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


def embed(x, dimension):
    """
    Create a lagged matrix similar to R's embed function.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array (can be 1D or 2D)
    dimension : int
        Number of lags
        
    Returns:
    --------
    embedded : numpy.ndarray
        Embedded matrix
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    
    if dimension > n_rows:
        raise ValueError("dimension cannot be larger than number of rows")
    
    result_rows = n_rows - dimension + 1
    result_cols = n_cols * dimension
    
    result = np.zeros((result_rows, result_cols))
    
    for i in range(dimension):
        start_col = i * n_cols
        end_col = (i + 1) * n_cols
        result[:, start_col:end_col] = x[dimension - 1 - i:n_rows - i, :]
    
    return result


def calculate_bic(y, X, fitted_model):
    """
    Calculate BIC for a linear model.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Response variable
    X : numpy.ndarray
        Design matrix
    fitted_model : LinearRegression
        Fitted model
        
    Returns:
    --------
    bic : float
        Bayesian Information Criterion
    """
    n = len(y)
    y_pred = fitted_model.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    k = X.shape[1] + 1  # number of parameters (including intercept)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


def runAR(Y, indice, lag, type="fixed"):
    """
    Run autoregressive model.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (0-indexed in Python, 1-indexed in R)
    lag : int
        Number of lags
    type : str
        Type of model: "fixed" or "bic"
        
    Returns:
    --------
    dict with keys:
        - model: fitted model
        - pred: prediction
        - coef: coefficients
    """
    # Adjust index from R (1-indexed) to Python (0-indexed)
    indice = indice - 1
    
    Y2 = Y[:, indice].reshape(-1, 1)
    aux = embed(Y2, 4 + lag)
    y = aux[:, 0]
    
    # Remove first (ncol(Y2)*lag) columns
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        # Remove first (ncol(Y2)*(lag-1)) columns
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    if type == "fixed":
        model = LinearRegression()
        model.fit(X, y)
        coef = np.concatenate([[model.intercept_], model.coef_])
    
    elif type == "bic":
        bb = np.inf
        best_model = None
        ar_coef = None
        
        for i in range(1, X.shape[1] + 1):
            m = LinearRegression()
            m.fit(X[:, :i], y)
            crit = calculate_bic(y, X[:, :i], m)
            
            if crit < bb:
                bb = crit
                best_model = m
                ar_coef = np.concatenate([[m.intercept_], m.coef_])
        
        model = best_model
        coef = np.zeros(X.shape[1] + 1)
        coef[:len(ar_coef)] = ar_coef
    
    # Prediction: c(1, X.out) %*% coef
    pred = np.dot(np.concatenate([[1], X_out]), coef)
    
    return {"model": model, "pred": pred, "coef": coef}


def ar_rolling_window(Y, nprev, indice=1, lag=1, type="fixed"):
    """
    Rolling window autoregressive forecasting.
    
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
    type : str
        Type of model: "fixed" or "bic"
        
    Returns:
    --------
    dict with keys:
        - pred: predictions
        - coef: coefficients for each window
        - errors: RMSE and MAE
    """
    save_coef = np.full((nprev, 5), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        # NOTE: R code does NOT pass type to runAR here, always uses default "fixed"
        fact = runAR(Y_window, indice, lag)
        
        save_coef[nprev - i, :min(len(fact["coef"]), 5)] = fact["coef"][:5]
        save_pred[nprev - i, 0] = fact["pred"]
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    # Create prediction line with NAs at the beginning
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'AR({lag}) Forecasting - {type}')
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
