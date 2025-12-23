"""
Configuration file for inflation forecasting.
Edit this file to set your data path and parameters.
"""

# ============================================================
# DATA CONFIGURATION
# ============================================================

# Path to your data file (CSV or Excel)
DATA_PATH = "data/data.csv"

# ============================================================
# FORECASTING PARAMETERS
# ============================================================

# Number of out-of-sample predictions (rolling window forecasts)
NPREV = 132

# Forecast window sizes (lags/horizons to forecast)
# Examples: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for all horizons
#           [3, 6, 12] for selected horizons only
WINDOW_SIZES = [3, 6, 12]

