"""MRI Data IO Module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
import nibabel
import numpy as np
import numpy.typing as npt
import re
from typing import Optional

from .base import MRIData
from .orientation import data_reorientation


def load_mri_data(
    path: Path | str,
    dtype: type = np.float64,
    orient: bool = True,
) -> MRIData:
    suffix_regex = re.compile(r".+(?P<suffix>(\.nii(\.gz|)|\.mg(z|h)))")
    m = suffix_regex.match(Path(path).name)
    if (m is not None) and (m.groupdict()["suffix"] in (".nii", ".nii.gz")):
        mri = nibabel.nifti1.load(path)
    elif (m is not None) and (m.groupdict()["suffix"] in (".mgz", ".mgh")):
        mri = nibabel.freesurfer.mghformat.load(path)
    else:
        raise ValueError(f"Invalid suffix {path}, should be either '.nii', or '.mgz'")

    affine = mri.affine
    if affine is None:
        raise RuntimeError("MRI do not contain affine")

    data = np.asarray(mri.get_fdata("unchanged"), dtype=dtype)
    mri = MRIData(data=data, affine=affine)

    if orient:
        return data_reorientation(mri)
    else:
        return mri


def save_mri_data(
    mri: MRIData, path: Path, dtype: npt.DTypeLike, intent_code: Optional[int] = None
):
    # TODO : Choose other way to check extension than regex ?
    suffix_regex = re.compile(r".+(?P<suffix>(\.nii(\.gz|)|\.mg(z|h)))")
    m = suffix_regex.match(Path(path).name)
    if (m is not None) and (m.groupdict()["suffix"] in (".nii", ".nii.gz")):
        nii = nibabel.nifti1.Nifti1Image(mri.data.astype(dtype), mri.affine)
        if intent_code is not None:
            nii.header.set_intent(intent_code)
        nibabel.nifti1.save(nii, path)
    elif (m is not None) and (m.groupdict()["suffix"] in (".mgz", ".mgh")):
        mgh = nibabel.freesurfer.mghformat.MGHImage(mri.data.astype(dtype), mri.affine)
        if intent_code is not None:
            mgh.header.set_intent(intent_code)
        nibabel.freesurfer.mghformat.save(mgh, path)
    else:
        raise ValueError(f"Invalid suffix {path}, should be either '.nii', or '.mgz'")
