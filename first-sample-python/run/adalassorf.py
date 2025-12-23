"""Adaptive LASSO + Random Forest run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_adalassorf import lasso_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running Adaptive LASSO + Random Forest models (PCE only)...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Run for PCE index only (as in R code)
adalassorf_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Adaptive LASSO RF lag {lag} (PCE)...")
    adalassorf_results[f'rf{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha=1, type="adalasso")

print("\nCombining results...")
# Combine predictions for PCE
pce = np.column_stack([adalassorf_results[f'rf{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/adalassorf-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Adaptive LASSO RF PCE lag {first_lag} RMSE: {adalassorf_results[f'rf{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("Adaptive LASSO + Random Forest processing complete!")
