# MRI IO - Test

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np
import pytest

from mritk.data import MRIData, load_mri_data, save_mri_data
from mritk.segmentation import Segmentation


def test_mri_io_nifti(tmp_path, mri_data_dir):
    input_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.nii.gz"

    output_file = tmp_path / "output_nifti.nii.gz"

    data, affine = load_mri_data(input_file, dtype=np.single)  ## TODO : Test orient=True case
    save_mri_data(data, affine, output_file)


def test_MRIData_io(tmp_path, mri_data_dir):
    input_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.nii.gz"

    output_file = tmp_path / "output_mridata.nii.gz"

    mri_data = MRIData.from_file(input_file)
    mri_data.save(output_file, intent_code=1006)


def test_MRIData_io_invalid_suffix(tmp_path, mri_data_dir):
    input_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.nii.gz"

    output_file = tmp_path / "output_mridata.invalid"

    mri_data = MRIData.from_file(input_file)
    try:
        mri_data.save(output_file, intent_code=1006)
        assert False, "Expected ValueError for invalid suffix"
    except ValueError as e:
        assert str(e) == f"Invalid suffix {output_file}, should be either '.nii', or '.mgz'"


def test_load_mri_data_invalid_suffix(mri_data_dir):
    input_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.invalid"
    try:
        load_mri_data(input_file)
        assert False, "Expected ValueError for invalid suffix"
    except ValueError as e:
        assert str(e) == f"Invalid suffix {input_file}, should be either '.nii', or '.mgz'"


@pytest.mark.parametrize("orient", (True, False))
def test_load_Segmentation(tmp_path, mri_data_dir, orient: bool):
    input_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    seg = Segmentation.from_file(input_file)
    assert seg.data.dtype == int
    mri = MRIData.from_file(input_file, dtype=np.single, orient=orient)
    output_file = tmp_path.with_suffix(".nii.gz")
    mri.save(output_file, dtype=np.single)
