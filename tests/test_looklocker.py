from pathlib import Path
from unittest.mock import patch

import numpy as np

import mritk.cli
from mritk.looklocker import (
    create_largest_island_mask,
    remove_outliers,
)


def test_looklocker_t1map(tmp_path, mri_data_dir: Path, gonzo_roi):
    LL_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1.nii.gz"
    timestamps = (
        mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1_trigger_times.txt"
    )
    T1_low = 100
    T1_high = 6000
    ll_file = mritk.data.MRIData.from_file(LL_path, dtype=np.single)
    vi = gonzo_roi.voxel_indices(affine=ll_file.affine)
    v = ll_file.data[tuple(vi.T)].reshape((*gonzo_roi.shape, -1))
    piece_ll_data = mritk.data.MRIData(data=v, affine=gonzo_roi.affine)
    ll_piece_path = Path("piece_ll.nii.gz")
    piece_ll_data.save(ll_piece_path)

    ll_data = mritk.looklocker.LookLocker.from_file(ll_piece_path, timestamps)
    t1_map = ll_data.t1_map()
    t1_post = t1_map.postprocess(T1_high=T1_high, T1_low=T1_low)

    t1_arr = t1_post.data

    ref_output = mri_data_dir / "mri-processed/mri_dataset/derivatives/sub-01/ses-01/sub-01_ses-01_acq-looklocker_T1map.nii.gz"
    ll_ref = mritk.data.MRIData.from_file(ref_output, dtype=np.single)
    v_ref = ll_ref.data[tuple(vi.T)].reshape((*gonzo_roi.shape,))
    # v_ref = mritk.looklocker.remove_outliers(v_ref, t1_low=T1_low, t1_high=T1_high)

    mritk.testing.compare_nifti_arrays(t1_arr, v_ref, data_tolerance=1e-12)


def test_remove_outliers():
    """Test that data is appropriately masked and clipped to physiological T1 bounds."""
    # 2x2x1 Mock Data
    data = np.array([[[10.0], [500.0]], [[1500.0], [8000.0]]])

    # Mask out the first element
    mask = np.array([[[False], [True]], [[True], [True]]])

    t1_low = 100.0
    t1_high = 2000.0

    result = remove_outliers(data, mask, t1_low, t1_high)

    # Expected:
    # [0,0,0] -> NaN (masked out)
    # [0,1,0] -> 500.0 (valid)
    # [1,0,0] -> 1500.0 (valid)
    # [1,1,0] -> NaN (exceeds t1_high)

    assert np.isnan(result[0, 0, 0])
    assert result[0, 1, 0] == 500.0
    assert result[1, 0, 0] == 1500.0
    assert np.isnan(result[1, 1, 0])


def test_create_largest_island_mask():
    """Test morphology logic identifies the primary body of data and ignores disconnected noise."""
    # Create a 15x15x15 empty space (3375 voxels, which is > 1000 so the background isn't
    # accidentally filled in by remove_small_holes)
    data = np.full((15, 15, 15), np.nan)

    # Create a large block in the center (Island 1)
    data[5:10, 5:10, 5:10] = 100.0

    # Create a tiny disconnected speck in the corner (Island 2)
    data[0, 0, 0] = 50.0

    # Run with small morphology radiuses
    mask = create_largest_island_mask(data, radius=1, erode_dilate_factor=1.0)

    # Speck should be dropped, major block should be True
    assert mask[0, 0, 0] == np.False_
    assert mask[7, 7, 7] == np.True_


@patch("mritk.looklocker.dicom_to_looklocker")
def test_dispatch_dcm2ll(mock_dicom_to_ll):
    """Test that dispatch correctly routes to dicom_to_looklocker."""

    mritk.cli.main(["looklocker", "dcm2ll", "-i", "dummy_in.dcm", "-o", "dummy_out.nii.gz"])

    mock_dicom_to_ll.assert_called_once_with(Path("dummy_in.dcm"), Path("dummy_out.nii.gz"))


@patch("mritk.looklocker.looklocker_t1map")
def test_dispatch_t1(mock_ll_t1map):
    """Test that dispatch correctly routes to looklocker_t1map."""

    mritk.cli.main(["looklocker", "t1", "-i", "data.nii.gz", "-t", "times.txt", "-o", "t1map.nii.gz"])

    mock_ll_t1map.assert_called_once_with(Path("data.nii.gz"), Path("times.txt"), output=Path("t1map.nii.gz"))


@patch("mritk.looklocker.looklocker_t1map_postprocessing")
def test_dispatch_postprocess(mock_postprocessing):
    """Test that dispatch correctly routes to looklocker_t1map_postprocessing."""

    mritk.cli.main(
        [
            "looklocker",
            "postprocess",
            "-i",
            "raw_t1.nii.gz",
            "-o",
            "clean_t1.nii.gz",
            "--t1-low",
            "50.0",
            "--t1-high",
            "5000.0",
            "--radius",
            "5",
            "--erode-dilate-factor",
            "1.5",
        ]
    )

    mock_postprocessing.assert_called_once_with(
        T1map=Path("raw_t1.nii.gz"),
        T1_low=50.0,
        T1_high=5000.0,
        radius=5,
        erode_dilate_factor=1.5,
        output=Path("clean_t1.nii.gz"),
    )
