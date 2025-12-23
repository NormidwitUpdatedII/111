"""
Random Forest run script - Python version.
Converted from first-sample/run/rf.R with exact correspondence.
Runs Random Forest for lags 1-12 on both CPI and PCE indices.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_rf import rf_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running Random Forest models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Random Forest models
rf_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Random Forest lag {lag}...")
    rf_results[f'rf{lag}c'] = rf_rolling_window(Y, NPREV, 1, lag)
    rf_results[f'rf{lag}p'] = rf_rolling_window(Y, NPREV, 2, lag)

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([rf_results[f'rf{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([rf_results[f'rf{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/rf-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/rf-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Random Forest CPI lag {first_lag} RMSE: {rf_results[f'rf{first_lag}c']['errors']['rmse']:.4f}")
print(f"Random Forest PCE lag {first_lag} RMSE: {rf_results[f'rf{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Random Forest processing complete!")
