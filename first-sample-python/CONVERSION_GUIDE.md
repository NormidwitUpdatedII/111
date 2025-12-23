# R to Python Conversion Mapping

This document provides a detailed mapping of R functions to their Python equivalents.

## File Structure Correspondence

| R File | Python File | Status |
|--------|-------------|--------|
| `func-ar.R` | `func_ar.py` | ✅ Complete |
| `func-lasso.R` | `func_lasso.py` | ✅ Complete |
| `func-rf.R` | `func_rf.py` | ✅ Complete |
| `func-nn.R` | `func_nn.py` | ✅ Complete |
| `func-xgb.R` | `func_xgb.py` | ✅ Complete |
| `func-bag.R` | `func_bag.py` | ✅ Complete |
| `func-boosting.R` | `func_boosting.py` | ✅ Complete |
| `func-fact.R` | `func_fact.py` | ✅ Complete |
| `func-jn.R` | `func_jn.py` | ✅ Complete |
| `func-csr.R` | `func_csr.py` | ✅ Complete |
| `func-scad.R` | - | ⚠️ Advanced (see notes) |
| `func-ucsv.R` | - | ⚠️ Advanced (see notes) |
| `func-lbvar.R` | - | ⚠️ Advanced (see notes) |
| `func-tfact.R` | - | ⚠️ Advanced (see notes) |
| `func-polilasso.R` | - | ⚠️ Advanced (see notes) |
| `func-rfols.R` | - | ⚠️ Advanced (see notes) |
| `func-adalassorf.R` | - | ⚠️ Advanced (see notes) |

## Function Correspondence

### Autoregressive Models (func_ar.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runAR()` | `runAR()` | Exact correspondence |
| `ar.rolling.window()` | `ar_rolling_window()` | Snake case naming |
| `embed()` | `embed()` | Custom implementation |
| `lm()` | `LinearRegression()` | sklearn equivalent |
| `BIC()` | `calculate_bic()` | Custom implementation |

### LASSO Models (func_lasso.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runlasso()` | `runlasso()` | Exact correspondence |
| `lasso.rolling.window()` | `lasso_rolling_window()` | Snake case naming |
| `ic.glmnet()` | `ic_glmnet_bic()` | Custom with CV |
| `princomp()` | `PCA()` | sklearn equivalent |
| `runpols()` | `runpols()` | Post-LASSO OLS |
| `pols.rolling.window()` | `pols_rolling_window()` | Snake case naming |

### Random Forest (func_rf.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runrf()` | `runrf()` | Exact correspondence |
| `rf.rolling.window()` | `rf_rolling_window()` | Snake case naming |
| `randomForest()` | `RandomForestRegressor()` | sklearn equivalent |
| `importance()` | `feature_importances_` | sklearn attribute |

### Neural Networks (func_nn.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runnn()` | `runnn()` | Exact correspondence |
| `nn.rolling.window()` | `nn_rolling_window()` | Snake case naming |
| `h2o.deeplearning()` | `keras.Sequential()` | TensorFlow/Keras |
| Architecture: c(32,16,8) | layers.Dense(32,16,8) | Same architecture |
| activation='Rectifier' | activation='relu' | Same activation |

### XGBoost (func_xgb.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runxgb()` | `runxgb()` | Exact correspondence |
| `xgb.rolling.window()` | `xgb_rolling_window()` | Snake case naming |
| `xgboost()` | `xgb.train()` | Python xgboost |
| Parameters | Same parameters | eta=0.05, max_depth=4, etc. |

### Bagging (func_bag.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runbagg()` | `runbagg()` | Simplified implementation |
| `bagg.rolling.window()` | `bagg_rolling_window()` | Snake case naming |
| `HDeconometrics::bagging()` | `bagging_pretesting()` | Custom implementation |

### Boosting (func_boosting.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runboost()` | `runboost()` | Exact correspondence |
| `boosting.rolling.window()` | `boosting_rolling_window()` | Snake case naming |
| `boosting()` | `boosting()` | Custom gradient boosting |

### Factor Models (func_fact.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runfact()` | `runfact()` | Exact correspondence |
| `fact.rolling.window()` | `fact_rolling_window()` | Snake case naming |

### Jackknife (func_jn.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runjn()` | `runjn()` | Simplified implementation |
| `jackknife.rolling.window()` | `jackknife_rolling_window()` | Snake case naming |
| `jackknife()` | `jackknife()` | Custom model combination |

### Complete Subset Regression (func_csr.py)

| R Function | Python Function | Notes |
|------------|-----------------|-------|
| `runcsr()` | `runcsr()` | Simplified implementation |
| `csr.rolling.window()` | `csr_rolling_window()` | Snake case naming |
| `HDeconometrics::csr()` | `csr_regression()` | Simplified version |

## Library Equivalents

| R Library | Python Library | Purpose |
|-----------|----------------|---------|
| `glmnet` | `scikit-learn` | LASSO/ElasticNet |
| `randomForest` | `scikit-learn` | Random Forest |
| `h2o` | `tensorflow/keras` | Neural Networks |
| `xgboost` | `xgboost` | Gradient Boosting |
| `boot` | `scikit-learn.utils` | Bootstrap sampling |
| Base R `lm()` | `sklearn.linear_model.LinearRegression` | Linear regression |
| Base R `princomp()` | `sklearn.decomposition.PCA` | PCA |
| `HDeconometrics` | Custom implementations | Specialized methods |

## Key Conversion Patterns

### 1. Indexing
```r
# R (1-indexed)
Y[,1]
X[1:10,]
```
```python
# Python (0-indexed)
Y[:, 0]
X[0:10, :]
```

### 2. Lists vs Dictionaries
```r
# R
result = list("pred"=pred, "coef"=coef)
```
```python
# Python
result = {"pred": pred, "coef": coef}
```

### 3. Matrix Operations
```r
# R
c(1, X.out) %*% coef
```
```python
# Python
np.dot(np.concatenate([[1], X_out]), coef)
```

### 4. Function Naming
```r
# R
ar.rolling.window()
```
```python
# Python
ar_rolling_window()
```

### 5. Data Structures
```r
# R
matrix(NA, nprev, 5)
```
```python
# Python
np.full((nprev, 5), np.nan)
```

## Testing Correspondence

To verify exact correspondence between R and Python implementations:

1. Use the same random seed
2. Load identical data
3. Compare predictions numerically
4. Check coefficients match
5. Verify error metrics (RMSE, MAE)

Example:
```r
# R
set.seed(42)
result = ar.rolling.window(Y, 10, 1, 1, "fixed")
```
```python
# Python
np.random.seed(42)
result = ar_rolling_window(Y, 10, 1, 1, "fixed")
```

## Advanced Methods Not Fully Implemented

Some methods require specialized libraries or complex MCMC implementations:

- **SCAD**: Smoothly Clipped Absolute Deviation penalty (would need ncvreg equivalent)
- **UCSV**: Unobserved Components Stochastic Volatility (requires MCMC sampler)
- **LBVAR**: Large Bayesian VAR (requires Bayesian inference framework)
- **TFACT**: Targeted factor models (requires pre-testing framework)
- **PoliLasso**: Polynomial LASSO with all interactions
- **RFols**: Random Forest with OLS on selected variables
- **AdaLassoRF**: Adaptive LASSO followed by Random Forest

For these methods, either:
1. Use the R implementation
2. Implement full Python versions using specialized libraries
3. Use the simplified versions provided (with reduced functionality)

## Performance Comparison

Expected performance characteristics:

| Model | R Speed | Python Speed | Notes |
|-------|---------|--------------|-------|
| AR | Fast | Fast | Similar |
| LASSO | Fast | Fast | Similar with sklearn |
| Random Forest | Medium | Medium | Similar |
| Neural Network | Medium | Medium | GPU can speed up Python |
| XGBoost | Fast | Fast | Similar |
| Boosting | Medium | Medium | Similar |

## Additional Notes

1. **Plotting**: Python uses matplotlib instead of R's base graphics
2. **Data saving**: CSV files use numpy.savetxt instead of write.table
3. **Progress indicators**: `cat()` in R becomes `print()` in Python
4. **Error handling**: Python is more explicit with type checking
5. **Memory management**: Python uses different memory model than R

## Validation

To validate the conversion:
1. Compare forecasts on sample data
2. Check error metrics match
3. Verify coefficient values
4. Test edge cases
5. Compare computation time

The implementations maintain algorithmic correspondence while adapting to Python conventions and libraries.
