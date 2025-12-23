"""
AR models run script - Python version.
Runs AR models for specified window sizes on both CPI and PCE indices.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_ar import ar_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running AR models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# Storage for results
ar_cpi = {}
ar_pce = {}
bar_cpi = {}
bar_pce = {}

# Run for each window size
for lag in WINDOW_SIZES:
    print(f"\nProcessing lag {lag}...")
    
    # Fixed AR models
    ar_cpi[lag] = ar_rolling_window(Y, NPREV, 1, lag, type="fixed")
    ar_pce[lag] = ar_rolling_window(Y, NPREV, 2, lag, type="fixed")
    
    # BIC AR models
    bar_cpi[lag] = ar_rolling_window(Y, NPREV, 1, lag, type="bic")
    bar_pce[lag] = ar_rolling_window(Y, NPREV, 2, lag, type="bic")

# Combine results
cpi = np.column_stack([ar_cpi[lag]["pred"] for lag in WINDOW_SIZES])
pce = np.column_stack([ar_pce[lag]["pred"] for lag in WINDOW_SIZES])
bcpi = np.column_stack([bar_cpi[lag]["pred"] for lag in WINDOW_SIZES])
bpce = np.column_stack([bar_pce[lag]["pred"] for lag in WINDOW_SIZES])

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/ar-cpi.csv", cpi, delimiter=";")
np.savetxt("forecasts/ar-pce.csv", pce, delimiter=";")
np.savetxt("forecasts/bicar-cpi.csv", bcpi, delimiter=";")
np.savetxt("forecasts/bicar-pce.csv", bpce, delimiter=";")

# Print errors
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
for lag in WINDOW_SIZES:
    print(f"Lag {lag}: AR-CPI RMSE={ar_cpi[lag]['errors']['rmse']:.4f}, "
          f"AR-PCE RMSE={ar_pce[lag]['errors']['rmse']:.4f}")

print("\nResults saved to forecasts/ directory")
