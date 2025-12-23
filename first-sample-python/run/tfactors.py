"""Targeted Factor Models run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_tfact import tfact_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running Targeted Factor models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Targeted Factor models
tfact_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Targeted Factor lag {lag}...")
    tfact_results[f'tfact{lag}c'] = tfact_rolling_window(Y, NPREV, 1, lag)
    tfact_results[f'tfact{lag}p'] = tfact_rolling_window(Y, NPREV, 2, lag)

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([tfact_results[f'tfact{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([tfact_results[f'tfact{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/tfact-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/tfact-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Targeted Factor CPI lag {first_lag} RMSE: {tfact_results[f'tfact{first_lag}c']['errors']['rmse']:.4f}")
print(f"Targeted Factor PCE lag {first_lag} RMSE: {tfact_results[f'tfact{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Targeted Factor processing complete!")
