"""Adaptive LASSO run script - Python version."""
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

print("Running Adaptive LASSO models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Adaptive LASSO models
adalasso_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Adaptive LASSO lag {lag}...")
    adalasso_results[f'adalasso{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="adalasso")
    adalasso_results[f'adalasso{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="adalasso")

print("\nRunning Post-Adaptive LASSO OLS models...")
print("=" * 50)

# Post-Adaptive LASSO OLS using selected variables from Adaptive LASSO
pols_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Post-Adaptive LASSO OLS lag {lag}...")
    pols_results[f'pols_adalasso{lag}c'] = pols_rolling_window(
        Y, NPREV, 1, lag, adalasso_results[f'adalasso{lag}c']['coef'])
    pols_results[f'pols_adalasso{lag}p'] = pols_rolling_window(
        Y, NPREV, 2, lag, adalasso_results[f'adalasso{lag}p']['coef'])

print("\nCombining results...")
# Combine Adaptive LASSO predictions for CPI and PCE
cpi = np.column_stack([adalasso_results[f'adalasso{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([adalasso_results[f'adalasso{i}p']['pred'] for i in WINDOW_SIZES])

# Combine Post-Adaptive LASSO OLS predictions for CPI and PCE
pols_cpi = np.column_stack([pols_results[f'pols_adalasso{i}c']['pred'] for i in WINDOW_SIZES])
pols_pce = np.column_stack([pols_results[f'pols_adalasso{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/adalasso-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/adalasso-pce.csv", pce, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-adalasso-cpi.csv", pols_cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-adalasso-pce.csv", pols_pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Adaptive LASSO CPI lag {first_lag} RMSE: {adalasso_results[f'adalasso{first_lag}c']['errors']['rmse']:.4f}")
print(f"Post-Adaptive LASSO OLS CPI lag {first_lag} RMSE: {pols_results[f'pols_adalasso{first_lag}c']['errors']['rmse']:.4f}")
print("=" * 50)
print("Adaptive LASSO and Post-Adaptive LASSO OLS processing complete!")
