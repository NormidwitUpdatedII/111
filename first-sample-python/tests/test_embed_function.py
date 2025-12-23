"""
Test the embed function to ensure it matches R's embed exactly.
This is CRITICAL as embed is used in almost all functions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functions.func_ar import embed


def test_embed_basic():
    """
    Test embed with simple case.
    R: embed(1:10, 3) produces:
    [1,] 3 2 1
    [2,] 4 3 2
    [3,] 5 4 3
    ...
    [8,] 10 9 8
    """
    x = np.arange(1, 11)  # 1 to 10
    result = embed(x, 3)
    
    print("Testing embed(1:10, 3)")
    print("Result shape:", result.shape)
    print("Expected: (8, 3)")
    print("First row:", result[0])
    print("Expected: [3, 2, 1]")
    print("Last row:", result[-1])
    print("Expected: [10, 9, 8]")
    
    # R's embed places most recent first, then older
    expected_first = [3, 2, 1]
    expected_last = [10, 9, 8]
    
    assert result.shape == (8, 3), f"Shape mismatch: {result.shape} vs (8, 3)"
    assert np.allclose(result[0], expected_first), f"First row: {result[0]} vs {expected_first}"
    assert np.allclose(result[-1], expected_last), f"Last row: {result[-1]} vs {expected_last}"
    
    print("✓ Basic embed test PASSED\n")


def test_embed_2d():
    """
    Test embed with 2D matrix.
    R: embed(matrix(1:12, ncol=2), 3)
    """
    # Create matrix like R's matrix(1:12, ncol=2)
    # R fills by column: col1 = 1,2,3,4,5,6; col2 = 7,8,9,10,11,12
    x = np.array([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])
    result = embed(x, 3)
    
    print("Testing embed(matrix(1:12, ncol=2), 3)")
    print("Input shape:", x.shape)
    print("Result shape:", result.shape)
    print("Expected: (4, 6)")
    
    # For 2D: each lag adds ncol columns
    # First row should be: [3,9,2,8,1,7] (most recent then older)
    print("First row:", result[0])
    print("Expected: [3, 9, 2, 8, 1, 7]")
    
    assert result.shape == (4, 6), f"Shape mismatch: {result.shape} vs (4, 6)"
    
    expected_first = [3, 9, 2, 8, 1, 7]
    assert np.allclose(result[0], expected_first), f"First row: {result[0]} vs {expected_first}"
    
    print("✓ 2D embed test PASSED\n")


def test_embed_dimension_4_plus_lag():
    """
    Test embed with dimension 4+lag (as used in AR function).
    """
    x = np.arange(1, 21).reshape(-1, 1)  # 20x1 matrix
    lag = 1
    dimension = 4 + lag  # = 5
    
    result = embed(x, dimension)
    
    print(f"Testing embed with lag={lag}, dimension=4+lag={dimension}")
    print("Input shape:", x.shape)
    print("Result shape:", result.shape)
    print("Expected rows:", 20 - 5 + 1, "= 16")
    print("Expected cols:", 1 * 5, "= 5")
    
    assert result.shape == (16, 5), f"Shape mismatch: {result.shape} vs (16, 5)"
    
    # First row: [5, 4, 3, 2, 1]
    print("First row:", result[0])
    print("Expected: [5, 4, 3, 2, 1]")
    assert np.allclose(result[0], [5, 4, 3, 2, 1]), "First row mismatch"
    
    print("✓ embed with 4+lag test PASSED\n")


def test_embed_for_ar():
    """
    Test the exact embed usage in AR function.
    """
    # Create sample data similar to actual inflation data
    np.random.seed(42)
    Y = np.random.randn(100, 10)  # 100 obs, 10 variables
    
    indice = 1  # First column (R uses 1-indexed)
    lag = 1
    
    Y2 = Y[:, indice - 1].reshape(-1, 1)
    aux = embed(Y2, 4 + lag)
    
    print("Testing embed usage in AR function")
    print("Y2 shape:", Y2.shape)
    print("aux shape:", aux.shape)
    print("Expected aux shape: (100 - 4, 5) = (96, 5)")
    
    y = aux[:, 0]  # Response variable
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, (n_cols_Y2 * lag):]  # Predictors
    
    print("y shape:", y.shape, "Expected: (96,)")
    print("X shape:", X.shape, "Expected: (96, 4)")
    
    assert aux.shape == (96, 5), f"aux shape: {aux.shape} vs (96, 5)"
    assert y.shape == (96,), f"y shape: {y.shape} vs (96,)"
    assert X.shape == (96, 4), f"X shape: {X.shape} vs (96, 4)"
    
    print("✓ AR embed usage test PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING EMBED FUNCTION")
    print("=" * 60)
    print()
    
    test_embed_basic()
    test_embed_2d()
    test_embed_dimension_4_plus_lag()
    test_embed_for_ar()
    
    print("=" * 60)
    print("ALL EMBED TESTS PASSED!")
    print("=" * 60)
