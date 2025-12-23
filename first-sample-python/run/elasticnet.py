"""Elastic Net run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_lasso import lasso_rolling_window, pols_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

alpha = 0.5  # Elastic Net: alpha=0.5 (balance between Ridge and LASSO)

print("Running Elastic Net models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Elastic Net models
elasticnet_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Elastic Net lag {lag}...")
    elasticnet_results[f'elasticnet{lag}c'] = lasso_rolling_window(Y, NPREV, 1, lag, alpha, type="lasso")
    elasticnet_results[f'elasticnet{lag}p'] = lasso_rolling_window(Y, NPREV, 2, lag, alpha, type="lasso")

print("\nRunning Post-Elastic Net OLS models...")
print("=" * 50)

# Post-Elastic Net OLS using selected variables from Elastic Net
pols_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing Post-Elastic Net OLS lag {lag}...")
    pols_results[f'pols_elasticnet{lag}c'] = pols_rolling_window(
        Y, NPREV, 1, lag, elasticnet_results[f'elasticnet{lag}c']['coef'])
    pols_results[f'pols_elasticnet{lag}p'] = pols_rolling_window(
        Y, NPREV, 2, lag, elasticnet_results[f'elasticnet{lag}p']['coef'])

print("\nCombining results...")
# Combine Elastic Net predictions for CPI and PCE
cpi = np.column_stack([elasticnet_results[f'elasticnet{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([elasticnet_results[f'elasticnet{i}p']['pred'] for i in WINDOW_SIZES])

# Combine Post-Elastic Net OLS predictions for CPI and PCE
pols_cpi = np.column_stack([pols_results[f'pols_elasticnet{i}c']['pred'] for i in WINDOW_SIZES])
pols_pce = np.column_stack([pols_results[f'pols_elasticnet{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/elasticnet-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/elasticnet-pce.csv", pce, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-elasticnet-cpi.csv", pols_cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/pols-elasticnet-pce.csv", pols_pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"Elastic Net CPI lag {first_lag} RMSE: {elasticnet_results[f'elasticnet{first_lag}c']['errors']['rmse']:.4f}")
print(f"Post-Elastic Net OLS CPI lag {first_lag} RMSE: {pols_results[f'pols_elasticnet{first_lag}c']['errors']['rmse']:.4f}")
print("=" * 50)
print("Elastic Net and Post-Elastic Net OLS processing complete!")
