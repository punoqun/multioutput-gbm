"""
Tests for multi-output optimization.
"""
import time
import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from pygbm import GradientBoostingRegressor


def test_multioutput_correctness_small():
    """Test that multi-output predictions are correct for small dataset."""
    X, y = make_regression(n_samples=100, n_features=10, n_targets=5, 
                          random_state=42, noise=0.1)
    
    gb = GradientBoostingRegressor(max_iter=50, learning_rate=0.1, 
                                   max_leaf_nodes=31, random_state=42)
    gb.fit(X, y)
    
    predictions = gb.predict_multi(X)
    
    # Check shape
    assert predictions.shape == (100, 5), f"Expected shape (100, 5), got {predictions.shape}"
    
    # Check that predictions are reasonable (not NaN, not Inf)
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN"
    assert not np.any(np.isinf(predictions)), "Predictions contain Inf"
    
    # Check that R2 score is reasonable (should be positive for training data)
    r2 = r2_score(y, predictions, multioutput='uniform_average')
    assert r2 > 0.5, f"R2 score too low: {r2}"
    
    print(f"✓ Small dataset test passed (R2: {r2:.4f})")


def test_multioutput_correctness_medium():
    """Test that multi-output predictions are correct for medium dataset."""
    X, y = make_regression(n_samples=500, n_features=20, n_targets=10, 
                          random_state=42, noise=1.0)
    
    gb = GradientBoostingRegressor(max_iter=20, learning_rate=0.1, 
                                   max_leaf_nodes=31, random_state=42)
    gb.fit(X, y)
    
    predictions = gb.predict_multi(X)
    
    # Check shape
    assert predictions.shape == (500, 10), f"Expected shape (500, 10), got {predictions.shape}"
    
    # Check that predictions are reasonable
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN"
    assert not np.any(np.isinf(predictions)), "Predictions contain Inf"
    
    # Check R2 score (lower threshold due to added noise)
    r2 = r2_score(y, predictions, multioutput='uniform_average')
    assert r2 > 0.4, f"R2 score too low: {r2}"
    
    print(f"✓ Medium dataset test passed (R2: {r2:.4f})")


def test_multioutput_prediction_consistency():
    """Test that predictions are consistent across multiple calls."""
    X, y = make_regression(n_samples=200, n_features=15, n_targets=8, 
                          random_state=42)
    
    gb = GradientBoostingRegressor(max_iter=10, random_state=42)
    gb.fit(X, y)
    
    # Make predictions multiple times
    pred1 = gb.predict_multi(X)
    pred2 = gb.predict_multi(X)
    pred3 = gb.predict_multi(X)
    
    # All predictions should be identical
    assert_allclose(pred1, pred2, rtol=1e-6, 
                   err_msg="Predictions differ between calls")
    assert_allclose(pred2, pred3, rtol=1e-6, 
                   err_msg="Predictions differ between calls")
    
    print("✓ Prediction consistency test passed")


def test_multioutput_binned_vs_numeric():
    """Test that binned and numeric predictions match."""
    X, y = make_regression(n_samples=200, n_features=15, n_targets=6, 
                          random_state=42)
    
    gb = GradientBoostingRegressor(max_iter=10, max_bins=255, random_state=42)
    gb.fit(X, y)
    
    # Get numeric predictions
    pred_numeric = gb.predict_multi(X)
    
    # Bin the data and predict
    X_binned = gb.bin_mapper_.transform(X)
    pred_binned = gb._raw_predict_multi(X_binned)
    
    # Predictions should be very close (small differences due to binning)
    assert_allclose(pred_numeric, pred_binned, rtol=1e-4, atol=1e-4,
                   err_msg="Binned and numeric predictions differ too much")
    
    print("✓ Binned vs numeric prediction test passed")


def test_multioutput_performance():
    """Test and report performance improvements for multi-output predictions."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Test with different sizes
    test_configs = [
        (500, 20, 5, 10),
        (1000, 30, 10, 20),
        (2000, 40, 15, 30),
    ]
    
    for n_samples, n_features, n_targets, max_iter in test_configs:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              n_targets=n_targets, random_state=42)
        
        print(f"\nDataset: {n_samples} samples, {n_features} features, {n_targets} targets")
        
        # Train
        start = time.time()
        gb = GradientBoostingRegressor(max_iter=max_iter, verbose=False, random_state=42)
        gb.fit(X, y)
        train_time = time.time() - start
        print(f"  Training time: {train_time:.3f}s")
        
        # Warmup (triggers numba compilation)
        _ = gb.predict_multi(X[:10])
        
        # Time predictions
        n_runs = 20
        start = time.time()
        for _ in range(n_runs):
            predictions = gb.predict_multi(X)
        pred_time = (time.time() - start) / n_runs
        
        print(f"  Prediction time (avg of {n_runs} runs): {pred_time:.4f}s")
        print(f"  Predictions per second: {n_samples / pred_time:.0f}")
        
        # Verify correctness
        r2 = r2_score(y, predictions, multioutput='uniform_average')
        print(f"  R2 score: {r2:.4f}")
    
    print("\n" + "="*60)


def test_single_output_still_works():
    """Ensure single-output regression still works after optimization."""
    X, y = make_regression(n_samples=200, n_features=10, n_targets=1, 
                          random_state=42)
    y = y.ravel()  # Single output
    
    gb = GradientBoostingRegressor(max_iter=50, random_state=42)
    gb.fit(X, y)
    
    predictions = gb.predict(X)
    
    # Check shape
    assert predictions.shape == (200,), f"Expected shape (200,), got {predictions.shape}"
    
    # Check correctness
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN"
    assert not np.any(np.isinf(predictions)), "Predictions contain Inf"
    
    r2 = r2_score(y, predictions)
    assert r2 > 0.5, f"R2 score too low: {r2}"
    
    print(f"✓ Single output test passed (R2: {r2:.4f})")


def test_multioutput_with_different_dimensions():
    """Test multi-output with various output dimensions."""
    for n_targets in [2, 5, 10, 20, 50]:
        X, y = make_regression(n_samples=200, n_features=15, n_targets=n_targets, 
                              random_state=42)
        
        gb = GradientBoostingRegressor(max_iter=50, random_state=42)
        gb.fit(X, y)
        
        predictions = gb.predict_multi(X)
        
        assert predictions.shape == (200, n_targets), \
            f"Wrong shape for {n_targets} targets: {predictions.shape}"
        
        r2 = r2_score(y, predictions, multioutput='uniform_average')
        # Lower threshold as it gets harder with more targets
        assert r2 > 0.2, f"R2 too low for {n_targets} targets: {r2}"
    
    print(f"✓ Variable dimensions test passed (tested {len([2, 5, 10, 20, 50])} dimensions)")


if __name__ == '__main__':
    print("Running multi-output optimization tests...\n")
    
    test_multioutput_correctness_small()
    test_multioutput_correctness_medium()
    test_multioutput_prediction_consistency()
    test_multioutput_binned_vs_numeric()
    test_single_output_still_works()
    test_multioutput_with_different_dimensions()
    test_multioutput_performance()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
