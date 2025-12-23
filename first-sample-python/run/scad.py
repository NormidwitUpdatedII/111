"""SCAD penalty regression run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_scad import scad_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running SCAD models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# SCAD models
scad_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing SCAD lag {lag}...")
    scad_results[f'scad{lag}c'] = scad_rolling_window(Y, NPREV, 1, lag)
    scad_results[f'scad{lag}p'] = scad_rolling_window(Y, NPREV, 2, lag)

print("\nCombining results...")
# Combine predictions for CPI and PCE
cpi = np.column_stack([scad_results[f'scad{i}c']['pred'] for i in WINDOW_SIZES])
pce = np.column_stack([scad_results[f'scad{i}p']['pred'] for i in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/scad-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/scad-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
first_lag = WINDOW_SIZES[0]
print(f"SCAD CPI lag {first_lag} RMSE: {scad_results[f'scad{first_lag}c']['errors']['rmse']:.4f}")
print(f"SCAD PCE lag {first_lag} RMSE: {scad_results[f'scad{first_lag}p']['errors']['rmse']:.4f}")
print("=" * 50)
print("SCAD processing complete!")
