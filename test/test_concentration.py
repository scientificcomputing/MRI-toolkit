"""MRI Concentration maps - Tests

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
import pytest
  
from mritk.concentration.concentration import ( 
    concentration
)

from mritk.t1_maps.utils import compare_nifti_images

def test_intracranial_concentration(tmp_path, mri_data_dir: Path):
    baseline_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-01_T1map_hybrid.nii.gz"
    sessions = range(2, 5)

    images_path = [
        mri_data_dir / f"mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-0{i}_T1map_hybrid.nii.gz" for i in sessions
    ]
    mask_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-intracranial_binary.nii.gz"
    r1 = 0.0032

    ref_outputs = [
      mri_data_dir / f"mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-0{i}_concentration.nii.gz" for i in sessions
    ]
    test_outputs = [  
      tmp_path / f"output_ses-0{i}_concentration.nii.gz" for i in sessions
    ]

    for i, s in enumerate(sessions):
        concentration(input=images_path[i], reference=baseline_path, output=test_outputs[i], r1=r1, mask=mask_path)
        compare_nifti_images(test_outputs[i], ref_outputs[i], data_tolerance=1e-4)