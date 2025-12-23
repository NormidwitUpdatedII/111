"""
Simple test for embed function - no external dependencies.
"""
import numpy as np


def embed(x, dimension):
    """
    Create a lagged matrix similar to R's embed function.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    
    if dimension > n_rows:
        raise ValueError("dimension cannot be larger than number of rows")
    
    result_rows = n_rows - dimension + 1
    result_cols = n_cols * dimension
    
    result = np.zeros((result_rows, result_cols))
    
    for i in range(dimension):
        start_col = i * n_cols
        end_col = (i + 1) * n_cols
        result[:, start_col:end_col] = x[dimension - 1 - i:n_rows - i, :]
    
    return result


print("=" * 60)
print("TESTING EMBED FUNCTION")
print("=" * 60)

# Test 1: Basic 1D
print("\nTest 1: embed(1:10, 3)")
x = np.arange(1, 11)
result = embed(x, 3)
print(f"Shape: {result.shape} (expected: (8, 3))")
print(f"First row: {result[0]} (expected: [3, 2, 1])")
print(f"Last row: {result[-1]} (expected: [10, 9, 8])")
assert result.shape == (8, 3), "FAILED"
assert np.allclose(result[0], [3, 2, 1]), "FAILED"
assert np.allclose(result[-1], [10, 9, 8]), "FAILED"
print("✓ PASSED")

# Test 2: 2D matrix
print("\nTest 2: embed(matrix(1:12, ncol=2), 3)")
x = np.array([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])
result = embed(x, 3)
print(f"Shape: {result.shape} (expected: (4, 6))")
print(f"First row: {result[0]} (expected: [3, 9, 2, 8, 1, 7])")
assert result.shape == (4, 6), "FAILED"
assert np.allclose(result[0], [3, 9, 2, 8, 1, 7]), "FAILED"
print("✓ PASSED")

# Test 3: dimension 4+lag (AR function pattern)
print("\nTest 3: embed for AR with lag=1, dimension=5")
x = np.arange(1, 21).reshape(-1, 1)
result = embed(x, 5)
print(f"Shape: {result.shape} (expected: (16, 5))")
print(f"First row: {result[0]} (expected: [5, 4, 3, 2, 1])")
print(f"Last row: {result[-1]} (expected: [20, 19, 18, 17, 16])")
assert result.shape == (16, 5), "FAILED"
assert np.allclose(result[0], [5, 4, 3, 2, 1]), "FAILED"
print("✓ PASSED")

# Test 4: dimension 4+lag for lag=12
print("\nTest 4: embed for AR with lag=12, dimension=16")
x = np.arange(1, 101).reshape(-1, 1)  # 100 observations
result = embed(x, 16)
print(f"Shape: {result.shape} (expected: (85, 16))")
assert result.shape == (85, 16), "FAILED"
# First row: [16, 15, 14, ..., 1]
expected_first = list(range(16, 0, -1))
assert np.allclose(result[0], expected_first), "FAILED"
print("✓ PASSED")

print("\n" + "=" * 60)
print("ALL EMBED TESTS PASSED!")
print("=" * 60)
