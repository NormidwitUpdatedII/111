"""Random Forest Factor (RFF) run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from functions.func_rffact import rffact_rolling_window
from functions.func_rf import rf_rolling_window
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

print("=" * 70)
print("RF-Factor Model (2-stage method)")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 70)

# Check if RF results exist, otherwise run RF first
rf_results_file = "forecasts/rf_importance.pkl"

if os.path.exists(rf_results_file):
    print("Loading pre-computed RF importance scores...")
    with open(rf_results_file, 'rb') as f:
        rf_importance = pickle.load(f)
else:
    print("RF importance scores not found. Running RF models first...")
    print("(This may take a while)")
    
    rf_importance = {'cpi': {}, 'pce': {}}
    
    for lag in WINDOW_SIZES:
        print(f"\nRunning RF lag {lag}...")
        rf_c = rf_rolling_window(Y, NPREV, 1, lag)
        rf_p = rf_rolling_window(Y, NPREV, 2, lag)
        
        rf_importance['cpi'][lag] = rf_c.get('save_importance', [])
        rf_importance['pce'][lag] = rf_p.get('save_importance', [])
    
    # Save RF importance for future use
    os.makedirs("forecasts", exist_ok=True)
    with open(rf_results_file, 'wb') as f:
        pickle.dump(rf_importance, f)
    print(f"\nRF importance scores saved to {rf_results_file}")

print("\nRunning RF-Factor models...")
print("=" * 70)

# Run RF-Factor for specified lags on both indices
results_cpi = []
results_pce = []

for lag in WINDOW_SIZES:
    print(f"\nProcessing lag {lag}...")
    
    # Get importance for this lag
    imp_cpi = rf_importance['cpi'].get(lag, None)
    imp_pce = rf_importance['pce'].get(lag, None)
    
    print(f"  - CPI (index 1)")
    rff_c = rffact_rolling_window(Y, NPREV, 1, lag, imp_cpi)
    results_cpi.append(rff_c["pred"])
    
    print(f"  - PCE (index 2)")
    rff_p = rffact_rolling_window(Y, NPREV, 2, lag, imp_pce)
    results_pce.append(rff_p["pred"])

# Combine results
cpi = np.column_stack(results_cpi)
pce = np.column_stack(results_pce)

# Save results (matching R output names: rffact-cpi.csv, rffact-pce.csv)
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/rffact-cpi.csv", cpi, delimiter=";")
np.savetxt("forecasts/rffact-pce.csv", pce, delimiter=";")

print("\n" + "=" * 70)
print("Results saved to forecasts/ directory")
print("  - forecasts/rffact-cpi.csv")
print("  - forecasts/rffact-pce.csv")
print("Done!")
