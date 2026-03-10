# MRI Concentration maps - Tests

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mritk.cli
from mritk.concentration import (
    compute_concentration_from_R1_array,
    compute_concentration_from_T1_array,
    concentration_from_R1_expr,
    concentration_from_T1,
    concentration_from_T1_expr,
)
from mritk.testing import compare_nifti_images


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
        concentration_from_T1(
            input_path=images_path[i],
            reference_path=baseline_path,
            output_path=test_outputs[i],
            r1=r1,
            mask_path=mask_path,
        )
        compare_nifti_images(test_outputs[i], ref_outputs[i], data_tolerance=1e-12)


def test_compute_concentration_array_no_mask():
    """Test the array computation correctly defaults to keeping all valid positive T1s when no mask is provided."""
    t1_data = np.array([1000.0, 1000.0])
    t10_data = np.array([2000.0, 2000.0])
    r1 = 0.005

    result = compute_concentration_from_T1_array(t1_data, t10_data, r1, mask=None)

    assert np.isclose(result[0], 0.1)
    assert np.isclose(result[1], 0.1)


def test_concentration_from_t1_expr():
    """Test the core math equation for T1-to-Concentration conversion."""
    t1 = np.array([1000.0])
    t1_0 = np.array([2000.0])
    r1 = 0.005

    # Math: C = (1 / 0.005) * ((1 / 1000) - (1 / 2000))
    #       C = 200 * (0.001 - 0.0005)
    #       C = 200 * 0.0005 = 0.1
    expected = np.array([0.1])

    result = concentration_from_T1_expr(t1, t1_0, r1)
    np.testing.assert_array_almost_equal(result, expected)


def test_concentration_from_r1_expr():
    """Test the core math equation for R1-to-Concentration conversion."""
    r1_map = np.array([2.0])
    r1_0_map = np.array([1.0])
    r1 = 0.005

    # Math: C = (1 / 0.005) * (2.0 - 1.0)
    #       C = 200 * 1.0 = 200.0
    expected = np.array([200.0])

    result = concentration_from_R1_expr(r1_map, r1_0_map, r1)
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_concentration_from_T1_array_masking():
    """Test that zero/negative/tiny T1 values and explicit masks yield NaNs."""
    t1_data = np.array([1000.0, 1e-12, 1000.0, 1000.0])
    t10_data = np.array([2000.0, 2000.0, 1e-12, 2000.0])

    # Explicit mask excluding the last voxel
    mask = np.array([True, True, True, False])
    r1 = 0.005

    result = compute_concentration_from_T1_array(t1_data, t10_data, r1, mask=mask)

    # Expectations:
    # Voxel 0: Valid, should be 0.1
    # Voxel 1: t1_data is <= 1e-10 -> NaN
    # Voxel 2: t10_data is <= 1e-10 -> NaN
    # Voxel 3: mask is False -> NaN

    assert np.isclose(result[0], 0.1)
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_compute_concentration_from_R1_array_masking():
    """Test that invalid R1 values (NaN/Inf) and explicit masks yield NaNs."""
    r1_data = np.array([2.0, np.nan, 2.0, 2.0])
    r10_data = np.array([1.0, 1.0, np.inf, 1.0])

    # Explicit mask excluding the last voxel
    mask = np.array([True, True, True, False])
    r1 = 0.005

    result = compute_concentration_from_R1_array(r1_data, r10_data, r1, mask=mask)

    # Expectations:
    # Voxel 0: Valid, should be 200.0
    # Voxel 1: r1_data is NaN -> NaN
    # Voxel 2: r10_data is Inf -> NaN
    # Voxel 3: mask is False -> NaN

    assert np.isclose(result[0], 200.0)
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_compute_concentration_from_R1_array_no_mask():
    """Test the array computation correctly defaults to keeping all finite R1s when no mask is provided."""
    r1_data = np.array([2.0, 2.0])
    r10_data = np.array([1.0, 1.0])
    r1 = 0.005

    result = compute_concentration_from_R1_array(r1_data, r10_data, r1, mask=None)

    assert np.isclose(result[0], 200.0)
    assert np.isclose(result[1], 200.0)


@patch("mritk.concentration.concentration_from_T1")
def test_dispatch_concentration_t1_defaults(mock_conc_t1):
    """Test the T1 concentration command with minimum required arguments."""
    mritk.cli.main(["concentration", "t1", "-i", "post_t1.nii.gz", "-r", "pre_t1.nii.gz"])

    # Verify paths are parsed and default arguments (r1, output, mask) are applied
    mock_conc_t1.assert_called_once_with(
        input_path=Path("post_t1.nii.gz"),
        reference_path=Path("pre_t1.nii.gz"),
        output_path=None,
        r1=0.0045,
        mask_path=None,
    )


@patch("mritk.concentration.concentration_from_T1")
def test_dispatch_concentration_t1_explicit(mock_conc_t1):
    """Test the T1 concentration command with all optional arguments explicitly provided."""
    mritk.cli.main(
        [
            "concentration",
            "t1",
            "--input",
            "post_t1.nii.gz",
            "--reference",
            "pre_t1.nii.gz",
            "--output",
            "concentration.nii.gz",
            "--r1",
            "0.005",
            "--mask",
            "brain_mask.nii.gz",
        ]
    )

    # Verify explicit overrides and type casting (e.g., float for r1)
    mock_conc_t1.assert_called_once_with(
        input_path=Path("post_t1.nii.gz"),
        reference_path=Path("pre_t1.nii.gz"),
        output_path=Path("concentration.nii.gz"),
        r1=0.005,
        mask_path=Path("brain_mask.nii.gz"),
    )


@patch("mritk.concentration.concentration_from_R1")
def test_dispatch_concentration_r1_defaults(mock_conc_r1):
    """Test the R1 concentration command with minimum required arguments."""
    mritk.cli.main(["concentration", "r1", "-i", "post_r1.nii.gz", "-r", "pre_r1.nii.gz"])

    # Verify paths are parsed and default arguments are applied
    mock_conc_r1.assert_called_once_with(
        input_path=Path("post_r1.nii.gz"), reference_path=Path("pre_r1.nii.gz"), output_path=None, r1=0.0045, mask_path=None
    )


@patch("mritk.concentration.concentration_from_R1")
def test_dispatch_concentration_r1_explicit(mock_conc_r1):
    """Test the R1 concentration command with all optional arguments explicitly provided."""
    mritk.cli.main(
        [
            "concentration",
            "r1",
            "--input",
            "post_r1.nii.gz",
            "--reference",
            "pre_r1.nii.gz",
            "--output",
            "conc_r1.nii.gz",
            "--r1",
            "0.0032",
            "--mask",
            "csf_mask.nii.gz",
        ]
    )

    # Verify explicit overrides and type casting
    mock_conc_r1.assert_called_once_with(
        input_path=Path("post_r1.nii.gz"),
        reference_path=Path("pre_r1.nii.gz"),
        output_path=Path("conc_r1.nii.gz"),
        r1=0.0032,
        mask_path=Path("csf_mask.nii.gz"),
    )
