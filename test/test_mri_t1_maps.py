"""MRI T1 maps - Tests

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
import pytest

from mritk.t1_maps.t1_maps import (
    looklocker_t1map,
    looklocker_t1map_postprocessing,
    mixed_t1map,
    mixed_t1map_postprocessing,
    hybrid_t1map,
)

from mritk.t1_maps.utils import compare_nifti_images


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


def test_mixed_t1map(tmp_path, mri_data_dir: Path):
    SE_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_SE-modulus.nii.gz"
    IR_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_IR-corrected-real.nii.gz"
    meta_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_meta.json"

    ref_output = mri_data_dir / "mri-dataset/mri_dataset/derivatives/sub-01" / "ses-01/sub-01_ses-01_acq-mixed_T1map.nii.gz"
    test_output_raw = tmp_path / "output_acq-mixed_T1map_raw.nii.gz"
    test_output = tmp_path / "output_acq-mixed_T1map.nii.gz"

    T1_low = 100
    T1_high = 10000

    mixed_t1map(
        SE_nii_path=SE_path, IR_nii_path=IR_path, meta_path=meta_path, T1_low=T1_low, T1_high=T1_high, output=test_output_raw
    )
    mixed_t1map_postprocessing(SE_nii_path=SE_path, T1_path=test_output_raw, output=test_output)

    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)


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
