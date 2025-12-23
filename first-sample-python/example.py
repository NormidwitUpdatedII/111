"""
Quick start example for using the Python forecasting library.
This script demonstrates basic usage of the converted R functions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from functions.func_ar import ar_rolling_window
from functions.func_lasso import lasso_rolling_window
from functions.func_rf import rf_rolling_window
from data_utils import load_rda_data

def main():
    print("=" * 70)
    print("Forecasting Inflation - Python Version - Quick Start Example")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    try:
        # Update this path to match your data location
        Y = load_rda_data("rawdata.rda")
        print(f"   Data loaded successfully: {Y.shape[0]} observations, {Y.shape[1]} variables")
    except FileNotFoundError:
        print("   Error: rawdata.rda not found!")
        print("   Please ensure the data file is in the first-sample-python directory")
        print("   or update the path in this script.")
        return
    
    # Set parameters
    nprev = 10  # Small number for quick testing (use 132 for full analysis)
    indice = 1  # First variable (CPI)
    lag = 1
    
    print(f"\n2. Running forecasts with nprev={nprev}, lag={lag}...")
    print("   (Using small nprev for quick testing)\n")
    
    # AR Model
    print("   a) Autoregressive (AR) model...")
    ar_result = ar_rolling_window(Y, nprev, indice, lag, type="fixed")
    print(f"      RMSE: {ar_result['errors']['rmse']:.6f}")
    print(f"      MAE:  {ar_result['errors']['mae']:.6f}")
    
    # LASSO Model
    print("\n   b) LASSO model...")
    lasso_result = lasso_rolling_window(Y, nprev, indice, lag, alpha=1, type="lasso")
    print(f"      RMSE: {lasso_result['errors']['rmse']:.6f}")
    print(f"      MAE:  {lasso_result['errors']['mae']:.6f}")
    
    # Random Forest Model
    print("\n   c) Random Forest model...")
    rf_result = rf_rolling_window(Y, nprev, indice, lag)
    print(f"      RMSE: {rf_result['errors']['rmse']:.6f}")
    print(f"      MAE:  {rf_result['errors']['mae']:.6f}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("3. Summary Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15}")
    print("-" * 70)
    print(f"{'AR (fixed)':<20} {ar_result['errors']['rmse']:<15.6f} {ar_result['errors']['mae']:<15.6f}")
    print(f"{'LASSO':<20} {lasso_result['errors']['rmse']:<15.6f} {lasso_result['errors']['mae']:<15.6f}")
    print(f"{'Random Forest':<20} {rf_result['errors']['rmse']:<15.6f} {rf_result['errors']['mae']:<15.6f}")
    print("=" * 70)
    
    print("\n4. Next Steps:")
    print("   - Run full analysis with nprev=132")
    print("   - Try different models (XGBoost, Neural Networks, etc.)")
    print("   - Explore run/ directory for complete examples")
    print("   - See README.md for detailed documentation")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
