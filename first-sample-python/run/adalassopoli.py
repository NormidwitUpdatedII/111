"""Adaptive Polynomial LASSO run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_polilasso import lasso_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

alpha = 1

print("Running Adaptive Polynomial LASSO models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Adaptive LASSO with polynomial features
adalassopoli_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Adaptive Polynomial LASSO lag {lag}...")
    adalassopoli_results[f'adalasso{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="adalasso")
    adalassopoli_results[f'adalasso{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="adalasso")

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([adalassopoli_results[f'adalasso{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([adalassopoli_results[f'adalasso{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/adalassopoli-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/adalassopoli-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Adaptive Polynomial LASSO CPI lag {first_lag} RMSE: {adalassopoli_results[f'adalasso{first_lag}c']['errors']['rmse']:.4f}")
print(f"Adaptive Polynomial LASSO PCE lag {first_lag} RMSE: {adalassopoli_results[f'adalasso{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Adaptive Polynomial LASSO processing complete!")
