import numpy as np
import pytest

from mritk.data.base import MRIData
from mritk.t1_maps.t1_to_r1 import compute_r1_array, convert_T1_to_R1, T1_to_R1


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

    r1_mri = convert_T1_to_R1(mri, scale=1000.0)

    expected_r1 = np.array([[[1.0, 0.5]]])

    np.testing.assert_array_almost_equal(r1_mri.data, expected_r1)
    np.testing.assert_array_equal(r1_mri.affine, affine)


def test_t1_to_r1_invalid_input():
    """Test the wrapper function throws ValueError on an invalid type input."""
    with pytest.raises(ValueError, match="Input should be a Path or MRIData"):
        # Explicitly passing a raw string instead of Path/MRIData
        T1_to_R1(input_mri="not_a_path_or_mridata")
