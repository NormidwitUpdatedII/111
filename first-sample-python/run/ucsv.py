"""Unobserved Components Stochastic Volatility run script - Python version."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_ucsv import ucsv_rw
from data_utils import load_data
from config import DATA_PATH, NPREV, WINDOW_SIZES

# Load data
Y = load_data(DATA_PATH)

# Use only CPI and PCE columns
Y = Y[:, :2]

print("Running UCSV models...")
print(f"Window sizes: {WINDOW_SIZES}")
print("=" * 50)

# UCSV forecasting for CPI
print("Processing UCSV for CPI...")
ucsv_cpi = ucsv_rw(100 * Y[:, 0], NPREV, h=WINDOW_SIZES) / 100

# UCSV forecasting for PCE
print("Processing UCSV for PCE...")
ucsv_pce = ucsv_rw(100 * Y[:, 1], NPREV, h=WINDOW_SIZES) / 100

# Save results
os.makedirs("forecasts", exist_ok=True)
np.savetxt("forecasts/ucsv-cpi.csv", ucsv_cpi, delimiter=";", fmt='%.6f')
np.savetxt("forecasts/ucsv-pce.csv", ucsv_pce, delimiter=";", fmt='%.6f')

print("\nResults saved to forecasts/ directory")
print("=" * 50)
print("UCSV processing complete!")
