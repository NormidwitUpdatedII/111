"""Random Forest OLS run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_rfols import rfols_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

# Set seed for reproducibility
np.random.seed(1)

print("Running Random Forest OLS models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# RF-OLS models
rfols_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing RF-OLS lag {lag}...")
    rfols_results[f'rf{lag}c'] = rfols_rolling_window(Y, NPREV, 1, lag)
    rfols_results[f'rf{lag}p'] = rfols_rolling_window(Y, NPREV, 2, lag)

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([rfols_results[f'rf{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([rfols_results[f'rf{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/rfols-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/rfols-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"RF-OLS CPI lag {first_lag} RMSE: {rfols_results[f'rf{first_lag}c']['errors']['rmse']:.4f}")
print(f"RF-OLS PCE lag {first_lag} RMSE: {rfols_results[f'rf{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Random Forest OLS processing complete!")
