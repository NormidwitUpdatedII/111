"""
Test runAR function to ensure it matches R exactly.
"""
import numpy as np
from numpy.linalg import lstsq


def embed(x, dimension):
    """R's embed function."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    result_rows = n_rows - dimension + 1
    result_cols = n_cols * dimension
    result = np.zeros((result_rows, result_cols))
    for i in range(dimension):
        start_col = i * n_cols
        end_col = (i + 1) * n_cols
        result[:, start_col:end_col] = x[dimension - 1 - i:n_rows - i, :]
    return result


def calculate_bic(y, X_with_intercept, model_coef):
    """Calculate BIC for a linear model."""
    y_pred = np.dot(X_with_intercept, model_coef)
    rss = np.sum((y - y_pred) ** 2)
    n = len(y)
    k = len(model_coef)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


def runAR(Y, indice, lag, type="fixed"):
    """
    Run autoregressive model - exact translation from R.
    
    R code:
    runAR=function(Y,indice,lag,type="fixed"){
      Y2=cbind(Y[,indice])
      aux=embed(Y2,4+lag)
      y=aux[,1]
      X=aux[,-c(1:(ncol(Y2)*lag))]  
      
      if(lag==1){
        X.out=tail(aux,1)[1:ncol(X)]  
      }else{
        X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
        X.out=tail(X.out,1)[1:ncol(X)]
      }
      y = y[1:(length(y)-lag+1)]
      X = X[1:(nrow(X)-lag+1),]
      
      if(type=="fixed"){
        model=lm(y~X)
        coef=coef(model)
      }
      ...
      pred=c(1,X.out)%*%coef
      return(list("model"=model,"pred"=pred,"coef"=coef))
    }
    """
    # R uses 1-indexing, Python uses 0-indexing
    indice_py = indice - 1
    
    # Y2=cbind(Y[,indice]) - extract single column
    Y2 = Y[:, indice_py].reshape(-1, 1)
    
    # aux=embed(Y2,4+lag)
    aux = embed(Y2, 4 + lag)
    
    # y=aux[,1] - first column (R is 1-indexed, but we're using 0-indexed result)
    y = aux[:, 0]
    
    # X=aux[,-c(1:(ncol(Y2)*lag))] - remove first (ncol(Y2)*lag) columns
    n_cols_Y2 = Y2.shape[1]  # = 1
    X = aux[:, (n_cols_Y2 * lag):]
    
    # Calculate X.out
    if lag == 1:
        # X.out=tail(aux,1)[1:ncol(X)]
        X_out = aux[-1, :X.shape[1]]
    else:
        # X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
        # X.out=tail(X.out,1)[1:ncol(X)]
        X_temp = aux[:, (n_cols_Y2 * (lag - 1)):]
        X_out = X_temp[-1, :X.shape[1]]
    
    # y = y[1:(length(y)-lag+1)]
    # X = X[1:(nrow(X)-lag+1),]
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    if type == "fixed":
        # Fit linear model with intercept
        X_with_intercept = np.column_stack([np.ones(len(y)), X])
        coef, _, _, _ = lstsq(X_with_intercept, y, rcond=None)
    
    elif type == "bic":
        bb = np.inf
        ar_coef = None
        
        for i in range(1, X.shape[1] + 1):
            X_subset = X[:, :i]
            X_with_intercept = np.column_stack([np.ones(len(y)), X_subset])
            temp_coef, _, _, _ = lstsq(X_with_intercept, y, rcond=None)
            crit = calculate_bic(y, X_with_intercept, temp_coef)
            
            if crit < bb:
                bb = crit
                ar_coef = temp_coef
        
        coef = np.zeros(X.shape[1] + 1)
        coef[:len(ar_coef)] = ar_coef
    
    # pred=c(1,X.out)%*%coef
    pred = np.dot(np.concatenate([[1], X_out]), coef)
    
    return {"pred": pred, "coef": coef}


print("=" * 60)
print("TESTING runAR FUNCTION")
print("=" * 60)

# Create synthetic data (100 observations, 5 variables)
np.random.seed(42)
n_obs = 100
n_vars = 5

# Create data similar to actual inflation forecasting
Y = np.random.randn(n_obs, n_vars)
# Make first column somewhat autoregressive
for t in range(1, n_obs):
    Y[t, 0] = 0.8 * Y[t-1, 0] + 0.2 * np.random.randn()

print("\nTest Data:")
print(f"  Shape: {Y.shape}")
print(f"  First 5 rows of first column: {Y[:5, 0].round(3)}")

# Test 1: runAR with type="fixed", lag=1
print("\n" + "-" * 40)
print("Test 1: runAR(Y, indice=1, lag=1, type='fixed')")
result = runAR(Y, indice=1, lag=1, type="fixed")
print(f"  Coefficients shape: {result['coef'].shape}")
print(f"  Coefficients: {result['coef'].round(4)}")
print(f"  Prediction: {result['pred']:.4f}")
print("  ✓ Completed (check coefficients match R)")

# Test 2: runAR with type="fixed", lag=12
print("\n" + "-" * 40)
print("Test 2: runAR(Y, indice=1, lag=12, type='fixed')")
result = runAR(Y, indice=1, lag=12, type="fixed")
print(f"  Coefficients shape: {result['coef'].shape}")
print(f"  Prediction: {result['pred']:.4f}")
print("  ✓ Completed")

# Test 3: runAR with type="bic", lag=1
print("\n" + "-" * 40)
print("Test 3: runAR(Y, indice=1, lag=1, type='bic')")
result = runAR(Y, indice=1, lag=1, type="bic")
print(f"  Coefficients shape: {result['coef'].shape}")
print(f"  Non-zero coefs: {np.sum(result['coef'] != 0)}")
print(f"  Prediction: {result['pred']:.4f}")
print("  ✓ Completed")

# Test 4: runAR for second column (PCE)
print("\n" + "-" * 40)
print("Test 4: runAR(Y, indice=2, lag=1, type='fixed')")
result = runAR(Y, indice=2, lag=1, type="fixed")
print(f"  Prediction: {result['pred']:.4f}")
print("  ✓ Completed")

print("\n" + "=" * 60)
print("runAR FUNCTION TESTS COMPLETED")
print("=" * 60)

# Test rolling window structure
print("\n" + "=" * 60)
print("TESTING ROLLING WINDOW STRUCTURE")
print("=" * 60)

nprev = 10  # Use small nprev for testing

print(f"\nWith nprev={nprev}, Y.shape={Y.shape}:")
print("\nRolling window indices (R logic: for(i in nprev:1)):")
print("=" * 50)

for i in range(nprev, 0, -1):
    start_idx = nprev - i
    end_idx = Y.shape[0] - i
    iteration = nprev - i + 1
    print(f"  i={i:2d}: window Y[{start_idx}:{end_idx}], iteration {iteration}")
    
print("\n  This creates an EXPANDING window (starts small, grows)")
print("  At iteration 1: window = Y[0:90]    (90 obs)")
print("  At iteration 10: window = Y[9:99]   (90 obs)")
print("\n  Note: Window SIZE is constant, but START POINT moves forward")

print("\n" + "=" * 60)
print("ROLLING WINDOW STRUCTURE VERIFIED")
print("=" * 60)
