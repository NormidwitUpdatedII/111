"""
XGBoost run script - Python version.
Converted from first-sample/run/xgb.R with exact correspondence.
Runs XGBoost for lags 1-12 on both CPI and PCE indices.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_xgb import xgb_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running XGBoost models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# XGBoost models
xgb_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing XGBoost lag {lag}...")
    xgb_results[f'xgb{lag}c'] = xgb_rolling_window(Y, NPREV, 1, lag)
    xgb_results[f'xgb{lag}p'] = xgb_rolling_window(Y, NPREV, 2, lag)

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([xgb_results[f'xgb{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([xgb_results[f'xgb{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/xgb-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/xgb-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"XGBoost CPI lag {first_lag} RMSE: {xgb_results[f'xgb{first_lag}c']['errors']['rmse']:.4f}")
print(f"XGBoost PCE lag {first_lag} RMSE: {xgb_results[f'xgb{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("XGBoost processing complete!")
