"""
Data loading utility for inflation forecasting.
Supports CSV, Excel, and pickle formats.
NO R DEPENDENCIES - fully standalone Python package.
"""
import numpy as np
import pandas as pd
import pickle
import os


def load_data(filepath):
    """
    Load data from various formats (CSV, Excel, pickle).
    
    Parameters:
    -----------
    filepath : str
        Path to the data file. Supported formats:
        - .csv (comma or semicolon separated)
        - .xlsx, .xls (Excel)
        - .pkl, .pickle (Python pickle)
        
    Returns:
    --------
    data : numpy.ndarray
        The loaded data as a numpy array
        
    Example:
    --------
    >>> Y = load_data("data/mydata.csv")
    >>> Y = load_data("data/mydata.xlsx")
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Try comma first, then semicolon
        try:
            data = pd.read_csv(filepath)
        except:
            data = pd.read_csv(filepath, sep=';')
    elif ext in ['.xlsx', '.xls']:
        data = pd.read_excel(filepath)
    elif ext in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                return data.values
            return data
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, or .pkl")
    
    # Convert to numpy array
    if isinstance(data, pd.DataFrame):
        return data.values
    return data


def load_rda_data(filepath):
    """
    DEPRECATED: Use load_data() instead.
    This function is kept for backward compatibility.
    
    If you have .rda files, first convert them to CSV using R:
        write.csv(dados, "data.csv", row.names=FALSE)
    
    Or install pyreadr: pip install pyreadr
    """
    # Check if file is actually CSV (for backward compatibility)
    if filepath.endswith('.csv'):
        return load_data(filepath)
    
    # Try to import pyreadr (optional dependency)
    try:
        import pyreadr
        result = pyreadr.read_r(filepath)
        key = list(result.keys())[0]
        data = result[key]
        if isinstance(data, pd.DataFrame):
            return data.values
        return data
    except ImportError:
        raise ImportError(
            "pyreadr is not installed. To load .rda files, either:\n"
            "1. Install pyreadr: pip install pyreadr\n"
            "2. Convert your .rda to CSV in R: write.csv(dados, 'data.csv', row.names=FALSE)\n"
            "3. Use load_data() with a CSV file instead"
        )


def save_data(data, filepath):
    """
    Save data to file (CSV or pickle).
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Data to save
    filepath : str
        Path to save the file (.csv or .pkl)
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    if ext == '.csv':
        data.to_csv(filepath, index=False, sep=';')
    elif ext in ['.pkl', '.pickle']:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        # Default to CSV
        data.to_csv(filepath, index=False, sep=';')


def save_data_pickle(data, filepath):
    """Save data to pickle format."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_data_pickle(filepath):
    """Load data from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_sample_data(n_rows=500, n_cols=125, seed=42):
    """
    Create sample data for testing.
    
    Parameters:
    -----------
    n_rows : int
        Number of time periods (default 500)
    n_cols : int
        Number of variables (default 125, matching original data)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Y : numpy.ndarray
        Sample data matrix where:
        - Column 0: CPI inflation (target 1)
        - Column 1: PCE inflation (target 2)
        - Columns 2+: Predictor variables
    """
    np.random.seed(seed)
    
    # Generate correlated time series
    Y = np.random.randn(n_rows, n_cols)
    
    # Add some autocorrelation to make it realistic
    for t in range(1, n_rows):
        Y[t, :] = 0.7 * Y[t-1, :] + 0.3 * Y[t, :]
    
    # Scale to realistic inflation values (around 2-4%)
    Y[:, 0] = Y[:, 0] * 0.5 + 2.5  # CPI
    Y[:, 1] = Y[:, 1] * 0.5 + 2.3  # PCE
    
    return Y
