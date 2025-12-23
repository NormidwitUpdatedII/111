"""Large Bayesian VAR run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_lbvar import lbvar_rw
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

p = 4  # VAR lag order

print("Running Large Bayesian VAR models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# LBVAR models
lbvar_results = {}
for lag in WINDOW_SIZES:
    print(f"Processing LBVAR lag {lag}...")
    lbvar_results[f'lbv{lag}'] = lbvar_rw(Y, p, lag, NPREV, variables=[1, 2])

print("\nCombining results...")
# Extract CPI and PCE predictions
cpi = np.zeros((NPREV, len(WINDOW_SIZES)))
pce = np.zeros((NPREV, len(WINDOW_SIZES)))

for idx, lag in enumerate(WINDOW_SIZES):
    pred = lbvar_results[f'lbv{lag}']['pred']
    cpi[:, idx] = pred[:, 0]
    pce[:, idx] = pred[:, 1]

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/lbvar-cpi.csv", cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/lbvar-pce.csv", pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
print("=" * 50)
print("Large Bayesian VAR processing complete!")
