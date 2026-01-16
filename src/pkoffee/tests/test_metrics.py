"""Tests for metrics module."""
import numpy as np
import pytest

from pkoffee.metrics import (
    SizeMismatchError,
    check_size_match,
    compute_r2,
    compute_rmse,
    compute_mae,
)

# ---------------------- Unit tests for check_size_match ----------------------

def test_check_size_match_success():
    """Test that check_size_match passes when arrays have same length."""
    array_a = np.array([1, 2, 3])
    array_b = np.array([4, 5, 6])
    check_size_match(array_a, array_b)

def test_size_mismatch_invalid():
    """Test that check_size_match raises an error when arrays have different lengths."""
    array_a = np.array([1, 2, 3])
    array_b = np.array([4, 5, 6, 7])
    with pytest.raises(expected_exception=SizeMismatchError, match="Arrays must have same length"):
        check_size_match(array_a, array_b)

# ----------------------- Unit tests for compute_r2 -----------------------------

def test_compute_r2_success():
    """Test that compute_r2 returns the correct R² score."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    r2 = compute_r2(y_true, y_pred)
    assert r2 == pytest.approx(0.9920)

def test_compute_r2_perfect_prediction():
    """Test that compute_r2 returns 1.0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    r2 = compute_r2(y_true, y_pred)
    assert r2 == pytest.approx(1.0)

def test_compute_r2_predicting_mean():
    """Test that compute_r2 returns 0.0 when predicting the mean."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.5, 2.5, 2.5, 2.5])
    r2 = compute_r2(y_true, y_pred)
    assert r2 == pytest.approx(0.0)

def test_compute_r2_negative():
    """Test that compute_r2 returns negative value when worse than mean."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, 10.0, 10.0, 10.0])  # very poor prediction
    r2 = compute_r2(y_true, y_pred)
    assert r2 < 0.0

def test_compute_r2_size_mismatch():
    """Test that compute_r2 raises SizeMismatchError for mismatched arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        compute_r2(y_true, y_pred)

# ----------------------- Unit tests for compute_rmse -----------------------------

def test_compute_rmse_success():
    """Test that compute_rmse returns the correct RMSE value."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    rmse = compute_rmse(y_true, y_pred)
    assert rmse == pytest.approx(0.1000)

def test_compute_rmse_perfect_prediction():
    """Test that compute_rmse returns 0.0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    rmse = compute_rmse(y_true, y_pred)
    assert rmse == pytest.approx(0.0)

def test_compute_rmse_negative_values():
    """Test compute_rmse with negative values."""
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([-1.1, -1.9, -3.1])
    rmse = compute_rmse(y_true, y_pred)
    # RMSE = sqrt(mean([0.1², 0.1², 0.1²])) = sqrt(mean([0.01, 0.01, 0.01])) = sqrt(0.01) = 0.1
    assert abs(rmse - 0.1) < 0.0001

def test_compute_rmse_size_mismatch():
    """Test that compute_rmse raises SizeMismatchError for mismatched arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        compute_rmse(y_true, y_pred)

# ----------------------- Unit tests for compute_mae -----------------------------

def test_compute_mae_success():
    """Test that compute_mae returns the correct MAE value."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    mae = compute_mae(y_true, y_pred)
    assert mae == pytest.approx(0.1000)

def test_compute_mae_perfect_prediction():
    """Test that compute_mae returns 0.0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    mae = compute_mae(y_true, y_pred)
    assert mae == pytest.approx(0.0)

def test_compute_mae_negative_values():
    """Test compute_mae with negative values."""
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([-1.1, -1.9, -3.1])
    mae = compute_mae(y_true, y_pred)
    # MAE = mean([0.1, 0.1, 0.1]) = 0.1
    assert abs(mae - 0.1) < 0.0001

def test_compute_mae_size_mismatch():
    """Test that compute_mae raises SizeMismatchError for mismatched arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        mae = compute_mae(y_true, y_pred)
        assert mae < 0.0