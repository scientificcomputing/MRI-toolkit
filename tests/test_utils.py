# Tests for T1 Map utilities

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


from pathlib import Path
from unittest.mock import patch

import numpy as np

from mritk.utils import (
    T1_lookup_table,
    estimate_se_free_relaxation_time,
    nan_filter_gaussian,
    run_dcm2niix,
    voxel_fit_function,
)


def test_voxel_fit_function():
    """Test the theoretical Look-Locker recovery curve math."""
    t = np.array([0.0, 1.0, 2.0])
    x1, x2, x3 = 1.0, 1.0, 1.0

    # Equation: abs(x1 * (1 - (1+x2^2)*exp(-x3^2 * t)))
    # With all 1s: abs(1 - 2*exp(-t))
    # t=0 -> abs(1 - 2*1) = 1
    # t=1 -> abs(1 - 2/e) ≈ 0.2642
    # t=2 -> abs(1 - 2/e^2) ≈ 0.7293

    expected = np.abs(1.0 - 2.0 * np.exp(-t))
    result = voxel_fit_function(t, x1, x2, x3)

    np.testing.assert_array_almost_equal(result, expected)


def test_nan_filter_gaussian():
    """Test that NaNs are smoothly interpolated without pulling valid data to zero."""
    # Create a 3x3 uniform array with a NaN hole in the center
    U = np.ones((3, 3))
    U[1, 1] = np.nan

    filtered = nan_filter_gaussian(U, sigma=1.0)

    # The NaN should be interpolated smoothly back to the surrounding value (1.0)
    assert not np.isnan(filtered[1, 1])
    np.testing.assert_array_almost_equal(filtered, np.ones((3, 3)))


def test_nan_filter_gaussian_edges():
    """Test the Gaussian filter handles edge NaNs gracefully."""
    U = np.ones((3, 3))
    U[0, 0] = np.nan  # Corner

    filtered = nan_filter_gaussian(U, sigma=1.0)
    assert not np.isnan(filtered[0, 0])
    np.testing.assert_array_almost_equal(filtered, np.ones((3, 3)))


def test_estimate_se_free_relaxation_time():
    """Test the calculation for free relaxation time."""
    TRse = 1000.0
    TE = 10.0
    ETL = 5

    # Formula check: TRse - TE * (1 + 0.5 * (ETL - 1) / (0.5 * (ETL + 1) + 20))
    # 1000 - 10 * (1 + 0.5 * 4 / (0.5 * 6 + 20))
    # 1000 - 10 * (1 + 2 / 23)
    expected = 1000.0 - 10.0 * (1.0 + 2.0 / 23.0)

    result = estimate_se_free_relaxation_time(TRse, TE, ETL)
    assert np.isclose(result, expected)


def test_t1_lookup_table():
    """Test the fraction/T1 lookup table generation creates arrays of correct shape/bounds."""
    TRse, TI, TE, ETL = 1000.0, 100.0, 10.0, 5
    T1_low, T1_hi = 100.0, 500.0

    fraction_curve, t1_grid = T1_lookup_table(TRse, TI, TE, ETL, T1_low, T1_hi)

    # Length should be exactly the integer steps from T1_low to T1_hi inclusive
    expected_length = int(T1_hi) - int(T1_low) + 1

    assert len(t1_grid) == expected_length
    assert len(fraction_curve) == expected_length
    assert t1_grid[0] == T1_low
    assert t1_grid[-1] == T1_hi

    # Check that fraction curve monotonically DECREASES for standard physics ranges
    # As T1 gets longer, the IR signal becomes more negative relative to the SE signal
    assert np.all(np.diff(fraction_curve) < 0)


@patch("subprocess.run")
def test_run_dcm2niix(mock_run):
    """Test that the dcm2niix command constructor triggers properly."""
    input_path = Path("/input/data.dcm")
    output_dir = Path("/output/")

    # Test valid execution
    run_dcm2niix(input_path, output_dir, form="test_form", extra_args="-z y")

    # Verify the constructed shell command
    mock_run.assert_called_once()
    args, _ = mock_run.call_args
    cmd = args[0]

    assert "dcm2niix" in cmd[0]
    assert "test_form" in cmd
    assert "-z" in cmd
    assert "y" in cmd
