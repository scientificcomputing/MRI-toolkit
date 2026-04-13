# Tests for T1 Map utilities

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


from pathlib import Path
from unittest.mock import patch

import numpy as np

from mritk.utils import (
    nan_filter_gaussian,
    run_dcm2niix,
)


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
