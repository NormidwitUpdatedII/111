"""
FINAL VERIFICATION SCRIPT
=========================
Run this script to verify the complete setup is working.
"""
import os
import sys

print("=" * 70)
print("INFLATION FORECASTING - SETUP VERIFICATION")
print("=" * 70)

# Check directory structure
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n[1] Base Directory: {base_dir}")

# Check required files
required_files = [
    "config.py",
    "data_utils.py",
    "requirements.txt",
    "README.md",
]

print("\n[2] Required Files:")
for f in required_files:
    path = os.path.join(base_dir, f)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"    {status} {f}")

# Check functions
functions_dir = os.path.join(base_dir, "functions")
function_files = [
    "func_ar.py", "func_lasso.py", "func_rf.py", "func_xgb.py",
    "func_nn.py", "func_boosting.py", "func_bag.py", "func_csr.py",
    "func_fact.py", "func_tfact.py", "func_jn.py", "func_lbvar.py",
    "func_scad.py", "func_ucsv.py", "func_rfols.py", "func_adalassorf.py",
    "func_polilasso.py", "func_rffact.py"
]

print(f"\n[3] Function Files ({len(function_files)} total):")
missing_funcs = []
for f in function_files:
    path = os.path.join(functions_dir, f)
    if not os.path.exists(path):
        missing_funcs.append(f)
        
if missing_funcs:
    for f in missing_funcs:
        print(f"    ✗ MISSING: {f}")
else:
    print(f"    ✓ All {len(function_files)} function files present")

# Check run scripts
run_dir = os.path.join(base_dir, "run")
run_files = [
    "ar.py", "lasso.py", "rf.py", "xgb.py", "nn.py", "boosting.py",
    "bagging.py", "csr.py", "factors.py", "tfactors.py", "jackknife.py",
    "lbvar.py", "scad.py", "ucsv.py", "rfols.py", "adalassorf.py",
    "adalasso.py", "adalassopoli.py", "adaelasticnet.py", "elasticnet.py",
    "ridge.py", "lassopoli.py", "rff.py"
]

print(f"\n[4] Run Scripts ({len(run_files)} total):")
missing_runs = []
for f in run_files:
    path = os.path.join(run_dir, f)
    if not os.path.exists(path):
        missing_runs.append(f)
        
if missing_runs:
    for f in missing_runs:
        print(f"    ✗ MISSING: {f}")
else:
    print(f"    ✓ All {len(run_files)} run scripts present")

# Check data directory
data_dir = os.path.join(base_dir, "data")
print(f"\n[5] Data Directory:")
if os.path.exists(data_dir):
    data_files = os.listdir(data_dir)
    print(f"    ✓ data/ directory exists")
    if data_files:
        for f in data_files:
            print(f"      - {f}")
    else:
        print("      (empty - add your data file here)")
else:
    print("    ✗ data/ directory missing")

# Check config
print(f"\n[6] Configuration (config.py):")
try:
    sys.path.insert(0, base_dir)
    from config import DATA_PATH, NPREV, ROLLING_WINDOW_TYPE, FORECAST_HORIZONS
    print(f"    DATA_PATH = {DATA_PATH}")
    print(f"    NPREV = {NPREV}")
    print(f"    ROLLING_WINDOW_TYPE = {ROLLING_WINDOW_TYPE}")
    print(f"    FORECAST_HORIZONS = {FORECAST_HORIZONS[:3]}...{FORECAST_HORIZONS[-1]}")
except Exception as e:
    print(f"    ✗ Error loading config: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_ok = not missing_funcs and not missing_runs
if all_ok:
    print("""
✓ All files are present and configuration is loaded.

NEXT STEPS:
1. Place your data file in the 'data/' folder
2. Edit config.py to set DATA_PATH to your data file
3. Run: python tests/test_core.py (to verify setup)
4. Run any model: python run/ar.py, python run/lasso.py, etc.

AVAILABLE MODELS (23):
  - AR (ar.py)              - LASSO (lasso.py)
  - Ridge (ridge.py)        - Elastic Net (elasticnet.py)  
  - Adaptive LASSO (adalasso.py)
  - Random Forest (rf.py)   - XGBoost (xgb.py)
  - Neural Network (nn.py)  - Boosting (boosting.py)
  - Bagging (bagging.py)    - CSR (csr.py)
  - Factors (factors.py)    - Targeted Factors (tfactors.py)
  - Jackknife (jackknife.py) - LBVAR (lbvar.py)
  - SCAD (scad.py)          - UCSV (ucsv.py)
  - RF-OLS (rfols.py)       - RF-Factor (rff.py)
  - Ada-LASSO Poly (adalassopoli.py)
  - Ada-LASSO RF (adalassorf.py)
  - LASSO Poly (lassopoli.py)
  - Ada-Elastic Net (adaelasticnet.py)
""")
else:
    print("\n✗ Some files are missing. Please check the output above.")
