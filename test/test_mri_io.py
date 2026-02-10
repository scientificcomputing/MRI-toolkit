"""MRI IO - Test

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
from mritk.data.io import load_mri_data, save_mri_data


def test_mri_io_nifti(tmp_path, mri_data_dir):
    input_file = mri_data_dir / "mri-processed/mri_dataset/derivatives/sub-01/ses-01/sub-01_ses-01_acq-mixed_T1map.nii.gz"

    output_file = tmp_path / "output_nifti.nii.gz"

    mri = load_mri_data(input_file, dtype=np.single, orient=False)  ## TODO : Test orient=True case
    save_mri_data(mri, output_file, dtype=np.single)
