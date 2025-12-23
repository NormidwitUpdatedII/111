"""Boosting run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_boosting import boosting_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running Boosting models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 70)

# Run boosting for specified lags on both indices
results_cpi = []
results_pce = []

for lag in WINDOW_SIZES:
    print(f"\nProcessing lag {lag}...")
    print(f"  - CPI (index 1)")
    boost_c = boosting_rolling_window(Y, NPREV, 1, lag)
    results_cpi.append(boost_c["pred"])
    
    print(f"  - PCE (index 2)")
    boost_p = boosting_rolling_window(Y, NPREV, 2, lag)
    results_pce.append(boost_p["pred"])

# Combine results
cpi = np.column_stack(results_cpi)
pce = np.column_stack(results_pce)

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/boosting-cpi.csv", cpi, delimiter=";")
np.savetxt("forecasts/boosting-pce.csv", pce, delimiter=";")

print("\n" + "=" * 70)
print("Results saved to forecasts/ directory")
print("Done!")
