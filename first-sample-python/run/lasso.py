"""
LASSO and Post-LASSO OLS run script - Python version.
Converted from first-sample/run/lasso.R with exact correspondence.
Runs LASSO and Post-LASSO OLS for lags 1-12 on both CPI and PCE indices.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_lasso import lasso_rolling_window, pols_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

alpha = 1

print("Running LASSO models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Standard LASSO models
lasso_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing lag {lag}...")
    lasso_results[f'lasso{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="lasso")
    lasso_results[f'lasso{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="lasso")

print("\nRunning Post-LASSO OLS models...")
print("=" * 50)

# Post-LASSO OLS using selected variables from LASSO
pols_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Post-LASSO OLS lag {lag}...")
    pols_results[f'pols_lasso{lag}c'] = pols_rolling_window(
        Y, NPREV, 1, lag, lasso_results[f'lasso{lag}c']['coef'])
    pols_results[f'pols_lasso{lag}p'] = pols_rolling_window(
        Y, NPREV, 2, lag, lasso_results[f'lasso{lag}p']['coef'])

print("\nCombining results...")
# Combine LASSO predictions for CPI and PCE
cpi = np.column_stack([lasso_results[f'lasso{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([lasso_results[f'lasso{i}p']['pred'] for i in WINDOW_SIZES])

# Combine Post-LASSO OLS predictions for CPI and PCE
pols_cpi = np.column_stack([pols_results[f'pols_lasso{i}c']['pred'] for i in WINDOW_SIZES])
pols_pce = np.column_stack([pols_results[f'pols_lasso{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/lasso-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/lasso-pce.csv", pce, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-lasso-cpi.csv", pols_cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-lasso-pce.csv", pols_pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"LASSO CPI lag {first_lag} RMSE: {lasso_results[f'lasso{first_lag}c']['errors']['rmse']:.4f}")
print(f"Post-LASSO OLS CPI lag {first_lag} RMSE: {pols_results[f'pols_lasso{first_lag}c']['errors']['rmse']:.4f}")
print("=" * 50)
print("LASSO and Post-LASSO OLS processing complete!")
