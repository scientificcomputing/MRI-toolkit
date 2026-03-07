from pathlib import Path

import numpy as np
import pytest

from mritk.t1.utils import compare_nifti_images
from mritk.t1.looklocker import (
    looklocker_t1map,
    looklocker_t1map_postprocessing,
    remove_outliers,
    create_largest_island_mask,
)


@pytest.mark.skip(reason="Takes too long")
def test_looklocker_t1map(tmp_path, mri_data_dir: Path):
    LL_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1.nii.gz"
    timestamps = (
        mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1_trigger_times.txt"
    )
    T1_low = 100
    T1_high = 6000

    ref_output = mri_data_dir / "mri-dataset/mri_dataset/derivatives/sub-01" / "ses-01/sub-01_ses-01_acq-looklocker_T1map.nii.gz"
    test_output_raw = tmp_path / "output_acq-looklocker_T1map_raw.nii.gz"
    test_output = tmp_path / "output_acq-looklocker_T1map.nii.gz"

    looklocker_t1map(looklocker_input=LL_path, timestamps=timestamps, output=test_output_raw)
    looklocker_t1map_postprocessing(T1map=test_output_raw, T1_low=T1_low, T1_high=T1_high, output=test_output)
    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)


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
