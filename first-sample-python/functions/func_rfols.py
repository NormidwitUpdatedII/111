"""
Random Forest OLS hybrid (RFOLS) for inflation forecasting.
Converted from first-sample/functions/func-rfols.R with exact correspondence.

This method uses Random Forest for variable selection, then applies OLS
on the selected variables from each tree.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.func_ar import embed


def runrfols(Y, indice, lag):
    """
    Run Random Forest + OLS hybrid model.
    Fits RF, extracts variables used in each tree, fits OLS on those variables,
    and averages predictions.
    
    Parameters:
    -----------
    Y : numpy.ndarray
        Data matrix
    indice : int
        Column index to forecast (1-indexed)
    lag : int
        Number of lags
        
    Returns:
    --------
    float
        Prediction
    """
    # Remove last column if it exists (as in R code)
    if Y.shape[1] > 2:
        Y = Y[:, :-1]
    
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
    
    # Fit Random Forest with specific parameters matching R
    model = RandomForestRegressor(
        n_estimators=500,
        max_leaf_nodes=25,  # maxnodes in R
        random_state=42,
        bootstrap=True
    )
    model.fit(X, y)
    
    # For each tree, get feature importances and fit OLS
    predaux = []
    
    for tree in model.estimators_:
        # Get features used in this tree
        feature_importances = tree.feature_importances_
        selected = np.where(feature_importances > 0)[0]
        
        if len(selected) == 0:
            # If no features selected, use all
            selected = np.arange(X.shape[1])
        
        # Fit OLS on selected features
        try:
            xaux = X[:, selected]
            modelols = LinearRegression()
            modelols.fit(xaux, y)
            
            # Predict
            pred_tree = modelols.predict(X_out[selected].reshape(1, -1))[0]
            predaux.append(pred_tree)
        except:
            # If OLS fails, use tree prediction
            pred_tree = tree.predict(X_out.reshape(1, -1))[0]
            predaux.append(pred_tree)
    
    # Average predictions across trees
    pred = np.mean(predaux)
    
    return pred


def rfols_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Random Forest OLS forecasting.
    
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
        - errors: RMSE and MAE
    """
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        pred = runrfols(Y_window, indice, lag)
        save_pred[nprev - i, 0] = pred
        print(f"iteration {nprev - i + 1}")
    
    # Adjust index for plotting
    real = Y[:, indice - 1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    pred_line = np.concatenate([np.full(len(real) - nprev, np.nan), save_pred.flatten()])
    plt.plot(pred_line, 'r-', label='Predicted')
    plt.legend()
    plt.title('Random Forest OLS Forecasting')
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
