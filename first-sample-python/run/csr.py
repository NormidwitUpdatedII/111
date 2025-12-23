"""Complete Subset Regression run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_csr import csr_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("Running Complete Subset Regression models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 70)

# Run CSR for specified lags on both indices
results_cpi = []
results_pce = []

for lag in WINDOW_SIZES:
    print(f"\nProcessing lag {lag}...")
    print(f"  - CPI (index 1)")
    csr_c = csr_rolling_window(Y, NPREV, 1, lag)
    results_cpi.append(csr_c["pred"])
    
    print(f"  - PCE (index 2)")
    csr_p = csr_rolling_window(Y, NPREV, 2, lag)
    results_pce.append(csr_p["pred"])

# Combine results
cpi = np.column_stack(results_cpi)
pce = np.column_stack(results_pce)

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/csr-cpi.csv", cpi, delimiter=";")
np.savetxt("forecasts/csr-pce.csv", pce, delimiter=";")

print("\n" + "=" * 70)
print("Results saved to forecasts/ directory")
print("Done!")
