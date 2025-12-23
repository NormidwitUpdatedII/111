"""Adaptive Elastic Net run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_lasso import lasso_rolling_window, pols_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

alpha = 0.5  # Adaptive Elastic Net: alpha=0.5 with adaptive weights

print("Running Adaptive Elastic Net models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Adaptive Elastic Net models
adaelasticnet_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Adaptive Elastic Net lag {lag}...")
    adaelasticnet_results[f'adaelasticnet{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="adalasso")
    adaelasticnet_results[f'adaelasticnet{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="adalasso")

print("\nRunning Post-Adaptive Elastic Net OLS models...")
print("=" * 50)

# Post-Adaptive Elastic Net OLS using selected variables
pols_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Post-Adaptive Elastic Net OLS lag {lag}...")
    pols_results[f'pols_adaelasticnet{lag}c'] = pols_rolling_window(
        Y, NPREV, 1, lag, adaelasticnet_results[f'adaelasticnet{lag}c']['coef'])
    pols_results[f'pols_adaelasticnet{lag}p'] = pols_rolling_window(
        Y, NPREV, 2, lag, adaelasticnet_results[f'adaelasticnet{lag}p']['coef'])

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([adaelasticnet_results[f'adaelasticnet{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([adaelasticnet_results[f'adaelasticnet{i}p']['pred'] for i in WINDOW_SIZES])

pols_cpi = np.column_stack([pols_results[f'pols_adaelasticnet{i}c']['pred'] for i in WINDOW_SIZES])
pols_pce = np.column_stack([pols_results[f'pols_adaelasticnet{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/adaelasticnet-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/adaelasticnet-pce.csv", pce, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-adaelasticnet-cpi.csv", pols_cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-adaelasticnet-pce.csv", pols_pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Adaptive Elastic Net CPI lag {first_lag} RMSE: {adaelasticnet_results[f'adaelasticnet{first_lag}c']['errors']['rmse']:.4f}")
print(f"Post-Adaptive Elastic Net OLS CPI lag {first_lag} RMSE: {pols_results[f'pols_adaelasticnet{first_lag}c']['errors']['rmse']:.4f}")
print("=" * 50)
print("Adaptive Elastic Net and Post-Adaptive Elastic Net OLS processing complete!")
