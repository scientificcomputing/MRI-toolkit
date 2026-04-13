from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import mritk.cli
from mritk.looklocker import create_largest_island_mask, remove_outliers, voxel_fit_function


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


@pytest.mark.xfail(
    reason=(
        "Generated T1 map does not match reference. "
        "Need to investigate whether this is a bug in the code "
        "or an issue with the test data."
    )
)
def _test_looklocker_t1map(tmp_path, mri_data_dir: Path, gonzo_roi):
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

    arr1 = np.nan_to_num(v_ref, nan=0.0)
    arr2 = np.nan_to_num(t1_arr, nan=0.0)

    worst_index = np.unravel_index(np.abs(arr1 - arr2).argmax(), arr1.shape)
    print(f"Worst voxel index: {worst_index}")
    print(f"Reference T1: {arr1[worst_index]}, Estimated T1: {arr2[worst_index]}")
    print(f"Unmasked Reference T1: {v_ref[worst_index]}, Unmasked Estimated T1: {t1_arr[worst_index]}")

    n_differences = np.sum(np.abs(arr1 - arr2) > 1e-12)
    print(
        f"Number of voxels with differences > 1e-12: {n_differences} out of {arr1.size} ({n_differences / arr1.size * 100:.2f}%)"
    )

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


@patch("mritk.looklocker.LookLocker")
def test_dispatch_t1(mock_ll):
    """Test that dispatch correctly routes to looklocker_t1map."""

    mritk.cli.main(["looklocker", "t1", "-i", "data.nii.gz", "-t", "times.txt", "-o", "t1map.nii.gz"])

    mock_ll.from_file.assert_called_once_with(Path("data.nii.gz"), Path("times.txt"))
    mock_ll.from_file.return_value.t1_map.assert_called_once()


@patch("mritk.looklocker.LookLockerT1")
def test_dispatch_postprocess(mock_ll_post):
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

    mock_ll_post.from_file.assert_called_once_with(Path("raw_t1.nii.gz"))
    inst = mock_ll_post.from_file.return_value
    inst.postprocess.assert_called_once_with(
        T1_low=50.0,
        T1_high=5000.0,
        radius=5,
        erode_dilate_factor=1.5,
    )
    inst.postprocess.return_value.save.assert_called_once_with(Path("clean_t1.nii.gz"), dtype=np.single)
