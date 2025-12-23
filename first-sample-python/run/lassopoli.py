"""Polynomial LASSO run script - Python version."""
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

print("Running Polynomial LASSO models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Polynomial LASSO models
lassopoli_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Polynomial LASSO lag {lag}...")
    lassopoli_results[f'lasso{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="lasso")
    lassopoli_results[f'lasso{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="lasso")

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([lassopoli_results[f'lasso{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([lassopoli_results[f'lasso{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/lassopoli-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/lassopoli-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Polynomial LASSO CPI lag {first_lag} RMSE: {lassopoli_results[f'lasso{first_lag}c']['errors']['rmse']:.4f}")
print(f"Polynomial LASSO PCE lag {first_lag} RMSE: {lassopoli_results[f'lasso{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Polynomial LASSO processing complete!")
