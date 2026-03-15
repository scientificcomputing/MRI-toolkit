# MRI R1 maps - Tests

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import mritk.cli
from mritk.data import MRIData
from mritk.r1 import (
    compute_r1_array,
    convert_t1_to_r1,
    t1_to_r1,
)


def test_compute_r1_array_standard():
    """Test basic T1 to R1 mathematical conversion."""
    t1_data = np.array([500.0, 1000.0, 2000.0])

    # Expected R1 = 1000 / T1
    expected = np.array([2.0, 1.0, 0.5])

    r1_data = compute_r1_array(t1_data, scale=1000.0)
    np.testing.assert_array_almost_equal(r1_data, expected)


def test_compute_r1_array_clipping():
    """Test that values outside the [t1_low, t1_high] bounds are safely set to NaN."""
    t1_data = np.array([0.5, 500.0, 6000.0, 10000.0])
    t1_low = 1.0
    t1_high = 5000.0

    r1_data = compute_r1_array(t1_data, scale=1000.0, t1_low=t1_low, t1_high=t1_high)

    # index 0 (0.5) < 1.0 -> NaN
    # index 1 (500) -> 2.0
    # index 2 (6000) > 5000.0 -> NaN
    # index 3 (10000) > 5000.0 -> NaN

    assert np.isnan(r1_data[0])
    assert r1_data[1] == 2.0
    assert np.isnan(r1_data[2])
    assert np.isnan(r1_data[3])


def test_convert_t1_to_r1_mridata():
    """Test the conversion properly preserves the MRIData class attributes (affine)."""
    t1_data = np.array([[[1000.0, 2000.0]]])
    affine = np.eye(4)
    mri = MRIData(data=t1_data, affine=affine)

    r1_mri = convert_t1_to_r1(mri, scale=1000.0)

    expected_r1 = np.array([[[1.0, 0.5]]])

    np.testing.assert_array_almost_equal(r1_mri.data, expected_r1)
    np.testing.assert_array_equal(r1_mri.affine, affine)


def test_t1_to_r1_invalid_input():
    """Test the wrapper function throws ValueError on an invalid type input."""
    with pytest.raises(ValueError, match="Input should be a Path or MRIData"):
        # Explicitly passing a raw string instead of Path/MRIData
        t1_to_r1(input_mri="not_a_path_or_mridata")


@patch("mritk.r1.t1_to_r1")
def test_dispatch_t1_to_r1_defaults(mock_t1_to_r1):
    """Test the T1 to R1 CLI command using default scaling and threshold values."""

    mritk.cli.main(["t1-to-r1", "-i", "input_t1.nii.gz", "-o", "output_r1.nii.gz"])

    # Verify the underlying function was called with parsed Paths and the correct defaults
    mock_t1_to_r1.assert_called_once_with(
        input_mri=Path("input_t1.nii.gz"),
        output=Path("output_r1.nii.gz"),
        scale=1000.0,  # Default value
        t1_low=1.0,  # Default value
        t1_high=float("inf"),  # Default value
    )


@patch("mritk.r1.t1_to_r1")
def test_dispatch_t1_to_r1_explicit_args(mock_t1_to_r1):
    """Test the T1 to R1 CLI command with all arguments explicitly provided."""

    mritk.cli.main(
        [
            "t1-to-r1",
            "--input",
            "input_t1.nii.gz",
            "--output",
            "output_r1.nii.gz",
            "--scale",
            "500.0",
            "--t1-low",
            "50.5",
            "--t1-high",
            "6000.0",
        ]
    )

    # Verify the underlying function received the explicit overrides and float conversions
    mock_t1_to_r1.assert_called_once_with(
        input_mri=Path("input_t1.nii.gz"), output=Path("output_r1.nii.gz"), scale=500.0, t1_low=50.5, t1_high=6000.0
    )


@patch("mritk.r1.t1_to_r1")
def test_dispatch_t1_to_r1_no_output(mock_t1_to_r1):
    """Test the T1 to R1 CLI command when the optional output argument is omitted."""

    mritk.cli.main(["t1-to-r1", "-i", "input_t1.nii.gz"])

    # Verify that output defaults to None when not provided
    mock_t1_to_r1.assert_called_once_with(
        input_mri=Path("input_t1.nii.gz"), output=None, scale=1000.0, t1_low=1.0, t1_high=float("inf")
    )
