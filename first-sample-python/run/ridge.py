"""Ridge regression run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_lasso import lasso_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

alpha = 0  # Ridge regression: alpha=0

print("Running Ridge models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Ridge models (Ridge is LASSO with alpha=0)
ridge_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Ridge lag {lag}...")
    ridge_results[f'ridge{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="lasso")
    ridge_results[f'ridge{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="lasso")

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([ridge_results[f'ridge{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([ridge_results[f'ridge{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/ridge-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/ridge-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Ridge CPI lag {first_lag} RMSE: {ridge_results[f'ridge{first_lag}c']['errors']['rmse']:.4f}")
print(f"Ridge PCE lag {first_lag} RMSE: {ridge_results[f'ridge{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Ridge processing complete!")
