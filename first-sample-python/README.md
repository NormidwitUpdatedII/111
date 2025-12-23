# Inflation Forecasting - Standalone Python Package

Python implementation of machine learning methods for inflation forecasting.  
**NO R DEPENDENCIES** - works directly with CSV/Excel files.

Based on "Forecasting Inflation in a data-rich environment: the benefits of machine learning methods" by Medeiros, Vasconcelos, Veiga, and Zilberman (2018).

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your data file in the `data/` folder:

```
data/
└── data.csv   (or .xlsx)
```

**Data Format:**
- Rows = time periods (observations)
- Column 1 = CPI inflation (or first target variable)
- Column 2 = PCE inflation (or second target variable)
- Columns 3+ = Predictor variables

Example:
```csv
CPI,PCE,Var1,Var2,Var3
2.1,1.8,0.5,1.2,3.4
2.3,1.9,0.6,1.1,3.2
...
```

### 3. Configure Settings

Edit `config.py`:
```python
DATA_PATH = "data/your_data.csv"  # Path to your data
NPREV = 132                        # Out-of-sample predictions
```

### 4. Run Models

```bash
cd run
python ar.py          # Autoregressive
python lasso.py       # LASSO
python rf.py          # Random Forest
python xgb.py         # XGBoost
python nn.py          # Neural Network
```

## Project Structure

```
first-sample-python/
├── config.py           # ← Edit this for your data path
├── data_utils.py       # Data loading (CSV, Excel, pickle)
├── requirements.txt    # Dependencies
├── README.md
├── data/               # ← Put your data here
│   └── data.csv
├── functions/          # Core algorithms (18 modules)
│   ├── func_ar.py
│   ├── func_lasso.py
│   ├── func_rf.py
│   └── ...
├── run/                # Run scripts (23 models)
│   ├── ar.py
│   ├── lasso.py
│   └── ...
└── forecasts/          # Output saved here
```

## Available Models (23 Total)

| Model | Script | Description |
|-------|--------|-------------|
| AR | `ar.py` | Autoregressive |
| LASSO | `lasso.py` | L1 regularization |
| Ridge | `ridge.py` | L2 regularization |
| Elastic Net | `elasticnet.py` | L1+L2 combined |
| Adaptive LASSO | `adalasso.py` | Weighted LASSO |
| Adaptive Elastic Net | `adaelasticnet.py` | Weighted Elastic Net |
| LASSO Polynomial | `lassopoli.py` | Polynomial features |
| Adaptive LASSO Poly | `adalassopoli.py` | Adaptive polynomial |
| Ada-LASSO + RF | `adalassorf.py` | Combined method |
| Random Forest | `rf.py` | RF regression |
| RF-OLS | `rfols.py` | RF selection + OLS |
| RF-Factor | `rff.py` | RF + Factor model |
| XGBoost | `xgb.py` | Gradient boosting |
| Neural Network | `nn.py` | Deep learning |
| Boosting | `boosting.py` | AdaBoost |
| Bagging | `bagging.py` | Bootstrap aggregating |
| Factors | `factors.py` | PCA factors |
| Targeted Factors | `tfactors.py` | Targeted PCA |
| CSR | `csr.py` | Complete subset |
| Jackknife | `jackknife.py` | Model combination |
| LBVAR | `lbvar.py` | Bayesian VAR |
| SCAD | `scad.py` | SCAD penalty |
| UCSV | `ucsv.py` | Stochastic volatility |

## Converting R Data (Optional)

If you have R `.rda` files, convert them in R first:
```r
load("rawdata.rda")
write.csv(dados, "data.csv", row.names = FALSE)
```

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn
- xgboost, tensorflow
- scipy, matplotlib

## Usage Examples

### Running Individual Models

```bash
# Autoregressive models
python run/ar.py

# LASSO models
python run/lasso.py

# Random Forest models
python run/rf.py

# XGBoost models
python run/xgb.py
```

### Using Functions Directly

You can also import and use the functions in your own scripts:

```python
import numpy as np
from functions.func_ar import ar_rolling_window
from data_utils import load_rda_data

# Load data
Y = load_rda_data("rawdata.rda")

# Run AR model with lag 1
results = ar_rolling_window(Y, nprev=132, indice=1, lag=1, type="fixed")

# Access predictions and errors
predictions = results["pred"]
rmse = results["errors"]["rmse"]
mae = results["errors"]["mae"]

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
```

## Key Functions

### Autoregressive Models (func_ar.py)
- `ar_rolling_window()`: Rolling window AR forecasting
  - Parameters: Y (data), nprev (forecast periods), indice (target column), lag, type ("fixed" or "bic")

### LASSO Models (func_lasso.py)
- `lasso_rolling_window()`: Rolling window LASSO
  - Parameters: Y, nprev, indice, lag, alpha (elastic net parameter), type ("lasso", "adalasso", "fal")
- `pols_rolling_window()`: Post-LASSO OLS

### Random Forest (func_rf.py)
- `rf_rolling_window()`: Rolling window Random Forest
  - Parameters: Y, nprev, indice, lag

### Neural Networks (func_nn.py)
- `nn_rolling_window()`: Rolling window neural network forecasting
  - Architecture: 3 hidden layers (32, 16, 8 neurons)
  - Activation: ReLU

### XGBoost (func_xgb.py)
- `xgb_rolling_window()`: Rolling window XGBoost
  - Parameters match R implementation (eta=0.05, max_depth=4, etc.)

### Other Methods
- `bagg_rolling_window()` - Bagging with pre-testing
- `boosting_rolling_window()` - Gradient boosting
- `fact_rolling_window()` - Factor models with BIC selection
- `jackknife_rolling_window()` - Jackknife model combination
- `csr_rolling_window()` - Complete Subset Regression

## Differences from R Implementation

### Core Algorithms
The Python implementations maintain exact correspondence with the R versions:
- Same data preprocessing (embedding, PCA)
- Same model parameters
- Same selection criteria (BIC, cross-validation)

### Libraries
Some R-specific libraries have been replaced with Python equivalents:
- `glmnet` → `scikit-learn` (LassoCV, ElasticNetCV)
- `randomForest` → `scikit-learn` (RandomForestRegressor)
- `h2o.deeplearning` → `tensorflow/keras`
- `xgboost` → `xgboost` (Python version)

### Advanced Methods
Some advanced methods from the HDeconometrics package have simplified implementations:
- CSR (Complete Subset Regression)
- Jackknife model combination
- Bagging with pre-testing

See `functions/README_ADVANCED_METHODS.md` for details on advanced method implementations.

## Output

Results are saved to the `forecasts/` directory as CSV files:
- `ar-cpi.csv`, `ar-pce.csv` - AR model predictions
- `lasso-cpi.csv`, `lasso-pce.csv` - LASSO predictions
- `rf-cpi.csv`, `rf-pce.csv` - Random Forest predictions
- `xgb-cpi.csv`, `xgb-pce.csv` - XGBoost predictions

Each file contains predictions for different lag specifications as columns.

## Performance Notes

- **Neural Networks**: Training can be slow for rolling windows. Consider reducing `epochs` or using GPU acceleration.
- **XGBoost**: 1000 rounds may be slow. Adjust `num_boost_round` if needed.
- **Random Forest**: 500 trees (default) can be memory intensive. Adjust `n_estimators` if needed.

## Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce `nprev` (number of forecast periods)
- Use fewer trees in Random Forest
- Reduce XGBoost rounds

### Import Errors
Make sure to run scripts from the `first-sample-python` directory:
```bash
cd first-sample-python
python run/ar.py
```

### Data Loading Issues
If `pyreadr` cannot load the `.rda` file, try:
1. Opening it in R and saving as `.RData`
2. Converting to CSV in R and loading with pandas

## References

Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). Forecasting Inflation in a data-rich environment: the benefits of machine learning methods.

## License

Same license as the original R repository (check LICENSE file in parent directory).

## Contact

For questions about the Python conversion, refer to the original paper for algorithmic details. For R implementation questions, see the original repository.

## Notes

- Paths in run scripts assume data is in the parent directory (`../rawdata.rda`)
- Modify paths as needed for your setup
- Random seeds are set for reproducibility where applicable
- Plotting is enabled by default; plots will appear during execution
