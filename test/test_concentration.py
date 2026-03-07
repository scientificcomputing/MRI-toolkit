# MRI Concentration maps - Tests

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

from pathlib import Path
import numpy as np
import pytest

from mritk.data.base import MRIData
from mritk.t1.concentration import (
    concentration,
    compute_r1_array,
    convert_T1_to_R1,
    T1_to_R1,
    concentration_from_T1,
    concentration_from_R1,
    compute_concentration_array,
)

from mritk.t1.utils import compare_nifti_images


def test_intracranial_concentration(tmp_path, mri_data_dir: Path):
    baseline_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-01_T1map_hybrid.nii.gz"
    sessions = [1, 2]

    images_path = [
        mri_data_dir / f"mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-0{i}_T1map_hybrid.nii.gz" for i in sessions
    ]
    mask_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-intracranial_binary.nii.gz"
    r1 = 0.0032

    ref_outputs = [
        mri_data_dir / f"mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-0{i}_concentration.nii.gz"
        for i in sessions
    ]
    test_outputs = [tmp_path / f"output_ses-0{i}_concentration.nii.gz" for i in sessions]

    for i, s in enumerate(sessions):
        concentration(
            input_path=images_path[i],
            reference_path=baseline_path,
            output_path=test_outputs[i],
            r1=r1,
            mask_path=mask_path,
        )
        compare_nifti_images(test_outputs[i], ref_outputs[i], data_tolerance=1e-12)


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


def test_concentration_from_t1():
    """Test the core math equation for T1-to-Concentration conversion."""
    t1 = np.array([1000.0])
    t1_0 = np.array([2000.0])
    r1 = 0.005

    # Math: C = (1 / 0.005) * ((1 / 1000) - (1 / 2000))
    #       C = 200 * (0.001 - 0.0005)
    #       C = 200 * 0.0005 = 0.1
    expected = np.array([0.1])

    result = concentration_from_T1(t1, t1_0, r1)
    np.testing.assert_array_almost_equal(result, expected)


def test_concentration_from_r1():
    """Test the core math equation for R1-to-Concentration conversion."""
    r1_map = np.array([2.0])
    r1_0_map = np.array([1.0])
    r1 = 0.005

    # Math: C = (1 / 0.005) * (2.0 - 1.0)
    #       C = 200 * 1.0 = 200.0
    expected = np.array([200.0])

    result = concentration_from_R1(r1_map, r1_0_map, r1)
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_concentration_array_masking():
    """Test that zero/negative/tiny values and explicit masks yield NaNs."""
    t1_data = np.array([1000.0, 1e-12, 1000.0, 1000.0])
    t10_data = np.array([2000.0, 2000.0, 1e-12, 2000.0])

    # Explicit mask excluding the last voxel
    mask = np.array([True, True, True, False])
    r1 = 0.005

    result = compute_concentration_array(t1_data, t10_data, r1, mask=mask)

    # Expectations:
    # Voxel 0: Valid, should be 0.1
    # Voxel 1: t1_data is <= 1e-10 -> NaN
    # Voxel 2: t10_data is <= 1e-10 -> NaN
    # Voxel 3: mask is False -> NaN

    assert np.isclose(result[0], 0.1)
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_compute_concentration_array_no_mask():
    """Test the array computation correctly defaults to keeping all valid positive T1s when no mask is provided."""
    t1_data = np.array([1000.0, 1000.0])
    t10_data = np.array([2000.0, 2000.0])
    r1 = 0.005

    result = compute_concentration_array(t1_data, t10_data, r1, mask=None)

    assert np.isclose(result[0], 0.1)
    assert np.isclose(result[1], 0.1)
