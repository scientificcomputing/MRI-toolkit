from pathlib import Path
from unittest.mock import patch

import numpy as np

import mritk.cli
from mritk.hybrid import compute_hybrid_t1_array, hybrid_t1map
from mritk.testing import compare_nifti_images


def test_hybrid_t1map(tmp_path, mri_data_dir: Path):
    LL_path = (
        mri_data_dir / "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_acq-looklocker_T1map_registered.nii.gz"
    )
    mixed_path = (
        mri_data_dir / "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_acq-mixed_T1map_registered.nii.gz"
    )
    csf_mask_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf_binary.nii.gz"
    test_output = tmp_path / "output_T1map_hybrid.nii.gz"
    ref_output = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-01_T1map_hybrid.nii.gz"
    threshold = 1500
    erode = 1

    hybrid_t1map(
        LL_path=LL_path, mixed_path=mixed_path, csf_mask_path=csf_mask_path, threshold=threshold, erode=erode, output=test_output
    )
    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)


def test_compute_hybrid_t1_array():
    """Test hybrid array logic merges LL and Mixed appropriately based on threshold and mask."""
    # 1D array for simplicity (4 voxels)
    ll_data = np.array([1000.0, 2000.0, 1000.0, 2000.0])
    mixed_data = np.array([500.0, 500.0, 3000.0, 3000.0])

    # Voxel 3 is unmasked
    mask = np.array([True, True, True, False])
    threshold = 1500.0

    hybrid = compute_hybrid_t1_array(ll_data, mixed_data, mask, threshold)

    # Evaluation: Substitution happens ONLY if BOTH > threshold AND inside mask.
    # Voxel 0: 1000 < 1500 -> Keep LL (1000.0)
    # Voxel 1: Mixed 500 < 1500 -> Keep LL (2000.0)
    # Voxel 2: LL (1000) < 1500 -> Keep LL (1000.0) ... wait, let's fix ll_data[2] to test properly
    # Let's run it as-is:
    assert hybrid[0] == 1000.0
    assert hybrid[1] == 2000.0
    assert hybrid[2] == 1000.0
    assert hybrid[3] == 2000.0  # Unmasked, so keep LL

    # Let's explicitly trigger the merge condition
    ll_data[2] = 2000.0
    hybrid2 = compute_hybrid_t1_array(ll_data, mixed_data, mask, threshold)
    # Voxel 2: LL(2000) > 1500 AND Mixed(3000) > 1500 AND Mask=True -> Merge!
    assert hybrid2[2] == 3000.0


@patch("mritk.hybrid.hybrid_t1map")
def test_dispatch_hybrid_defaults(mock_hybrid_t1map):
    """Test the hybrid CLI command with required arguments, relying on defaults for threshold and erode."""

    # We pass the arguments exactly as a user would type them in the terminal.
    # If "hybrid" is nested under another command (like "t1maps hybrid"), add that prefix to the list.
    mritk.cli.main(
        ["hybrid", "-l", "ll_map.nii.gz", "-m", "mixed_map.nii.gz", "-c", "csf_mask.nii.gz", "-o", "output_hybrid.nii.gz"]
    )

    # Verify the underlying function was called with parsed Paths and the correct default values
    mock_hybrid_t1map.assert_called_once_with(
        LL_path=Path("ll_map.nii.gz"),
        mixed_path=Path("mixed_map.nii.gz"),
        csf_mask_path=Path("csf_mask.nii.gz"),
        threshold=4000.0,  # Default value
        erode=0,  # Default value
        output=Path("output_hybrid.nii.gz"),
    )


@patch("mritk.hybrid.hybrid_t1map")
def test_dispatch_hybrid_explicit_args(mock_hybrid_t1map):
    """Test the hybrid CLI command with all arguments explicitly provided using long-form flags."""

    mritk.cli.main(
        [
            "hybrid",
            "--input-looklocker",
            "ll_map.nii.gz",
            "--input-mixed",
            "mixed_map.nii.gz",
            "--csf-mask",
            "csf_mask.nii.gz",
            "--threshold",
            "3500.5",
            "--erode",
            "2",
            "--output",
            "output_hybrid.nii.gz",
        ]
    )

    # Verify the underlying function received the explicit overrides and correct types
    mock_hybrid_t1map.assert_called_once_with(
        LL_path=Path("ll_map.nii.gz"),
        mixed_path=Path("mixed_map.nii.gz"),
        csf_mask_path=Path("csf_mask.nii.gz"),
        threshold=3500.5,
        erode=2,
        output=Path("output_hybrid.nii.gz"),
    )
