"""Tests for Masks and Intracranial modules

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

import mritk.cli
from mritk.masks import compute_csf_mask_array, compute_intracranial_mask_array, csf_mask, intracranial_mask, largest_island
from mritk.testing import compare_nifti_images


def test_largest_island():
    """Test that the largest contiguous region is correctly isolated."""
    mask = np.zeros((10, 10), dtype=bool)
    # Small island (4 pixels)
    mask[1:3, 1:3] = True
    # Large island (9 pixels)
    mask[5:8, 5:8] = True

    result = largest_island(mask, connectivity=1)

    # Large island should be kept
    assert result[6, 6] == np.True_
    # Small island should be dropped
    assert result[1, 1] == np.False_
    # Total active pixels should equal the large island's size
    assert np.sum(result) == 9


def test_largest_island_empty():
    """Test behavior when the mask contains no true values."""
    mask = np.zeros((5, 5), dtype=bool)
    result = largest_island(mask)

    assert np.sum(result) == 0
    assert result.dtype == bool


def test_compute_csf_mask_array_li():
    """Test generating a CSF mask using Li thresholding and largest island extraction."""
    # Create a 10x10x10 mock volume
    vol = np.zeros((10, 10, 10))
    # Add a main bright "CSF" island
    vol[2:8, 2:8, 2:8] = 100.0
    # Add a smaller disconnected "noise" island
    vol[0, 0, 0] = 100.0

    mask = compute_csf_mask_array(vol, connectivity=1, use_li=True)

    # Validates that the primary island is kept
    assert mask[5, 5, 5] == np.True_
    # Validates that the smaller noise island is dropped
    assert mask[0, 0, 0] == np.False_
    # Validates background is excluded
    assert mask[1, 1, 1] == np.False_


def test_compute_csf_mask_array_yen():
    """Test generating a CSF mask using Yen thresholding on histogram data."""
    # Ensure reproducible random distributions across Python versions
    np.random.seed(42)

    # Base background noise
    vol = np.random.uniform(1, 10, (15, 15, 15))

    # Inject primary bright island (e.g. CSF)
    # The uniform distribution guarantees different float values, meaning the
    # top 0.1% filter will only remove the 3 absolute brightest voxels.
    # The rest of the island will safely remain to populate the histogram.
    vol[4:10, 4:10, 4:10] = np.random.uniform(100, 150, (6, 6, 6))

    mask = compute_csf_mask_array(vol, connectivity=2, use_li=False)

    # Check that the center of the island is identified
    assert bool(mask[7, 7, 7]) is True
    # Check that the extreme background corners are completely excluded
    assert bool(mask[0, 0, 0]) is False
    assert bool(mask[14, 14, 14]) is False


def test_compute_intracranial_mask_array():
    """Test the array logic for merging CSF and segmentation into a solid bounding space."""
    # 10x10x10 empty background arrays
    csf = np.zeros((10, 10, 10), dtype=bool)
    seg = np.zeros((10, 10, 10), dtype=bool)

    # Outer ring = CSF
    csf[2:8, 2:8, 2:8] = True
    csf[3:7, 3:7, 3:7] = False

    # Inner core = Brain Segmentation
    seg[3:7, 3:7, 3:7] = True

    # Merge and smooth
    ic_mask = compute_intracranial_mask_array(csf, seg)

    # The whole 2:8 block should be completely solid (True)
    assert np.all(ic_mask[2:8, 2:8, 2:8] == np.True_)
    # Surrounding coordinates should remain background (False)
    assert ic_mask[0, 0, 0] == np.False_


def test_csf_mask_io(tmp_path):
    """Test the I/O wrapper for CSF masking by writing actual temporary NIfTI files."""
    in_path = tmp_path / "mock_in.nii.gz"
    out_path = tmp_path / "mock_out.nii.gz"

    # Create a real, small NIfTI file with an identity affine matrix
    data = np.zeros((10, 10, 10), dtype=np.single)
    data[2:8, 2:8, 2:8] = 100.0  # CSF target area
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, in_path)

    result = csf_mask(input=in_path, use_li=True)
    result.save(out_path, dtype=np.uint8)

    # Verify the file was physically saved to the filesystem
    assert out_path.exists()
    # Verify the output data shape matches what we expect
    assert result.data.shape == (10, 10, 10)


def test_intracranial_mask_io(tmp_path):
    """Test the I/O wrapper for Intracranial masking by writing actual temporary NIfTI files."""
    csf_path = tmp_path / "csf.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"
    out_path = tmp_path / "ic_out.nii.gz"

    # Create standard identity affine matrices to satisfy assert_same_space
    affine = np.eye(4)

    # 1. Mock CSF Mask file
    csf_data = np.zeros((10, 10, 10), dtype=np.single)
    csf_data[2:8, 2:8, 2:8] = 1.0
    nib.save(nib.Nifti1Image(csf_data, affine), csf_path)

    # 2. Mock Segmentation file
    seg_data = np.zeros((10, 10, 10), dtype=np.single)
    seg_data[4:6, 4:6, 4:6] = 1.0
    nib.save(nib.Nifti1Image(seg_data, affine), seg_path)

    result = intracranial_mask(csf_segmentation_path=csf_path, segmentation_path=seg_path)
    result.save(out_path, dtype=np.uint8)

    # Verify the file was physically saved to the filesystem
    assert out_path.exists()
    # Verify the output data shape matches what we expect
    assert result.data.shape == (10, 10, 10)


@patch("mritk.masks.csf_mask")
def test_dispatch_csf_mask(mock_csf_mask):
    """Test the CLI dispatch for the CSF mask command."""
    mritk.cli.main(["mask", "csf", "-i", "input.nii.gz", "--output", "mock_out.nii.gz", "--use-li", "--connectivity", "2"])

    mock_csf_mask.assert_called_once_with(input=Path("input.nii.gz"), connectivity=2, use_li=True)


@patch("mritk.masks.intracranial_mask")
def test_dispatch_intracranial_mask(mock_intracranial_mask):
    """Test the CLI dispatch for the intracranial mask command."""
    mritk.cli.main(
        [
            "mask",
            "intracranial",
            "--csf-segmentation-path",
            "csf_segmentation.nii.gz",
            "--segmentation-path",
            "segmentation.nii.gz",
            "-o",
            "ic_mask.nii.gz",
        ]
    )

    mock_intracranial_mask.assert_called_once_with(
        csf_segmentation_path=Path("csf_segmentation.nii.gz"), segmentation_path=Path("segmentation.nii.gz")
    )


def test_csf_mask(tmp_path, mri_data_dir: Path):
    input_T2w_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_T2w_registered.nii.gz"
    use_li = True

    ref_output = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf_binary.nii.gz"
    test_output = tmp_path / "output_seg-csf_binary.nii.gz"

    result = csf_mask(input=input_T2w_path, use_li=use_li)
    result.save(test_output, dtype=np.uint8)
    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)


def test_intracranial_mask(tmp_path, mri_data_dir: Path):
    csf_segmentation_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf-aseg.nii.gz"
    segmentation_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-wmparc_refined.nii.gz"

    ref_output = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-intracranial_binary.nii.gz"
    test_output = tmp_path / "output_seg-intracranial_binary.nii.gz"

    result = intracranial_mask(csf_segmentation_path=csf_segmentation_path, segmentation_path=segmentation_path)
    result.save(test_output, dtype=np.uint8)
    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)
