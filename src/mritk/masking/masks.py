"""Intracranial and CSF masks generation module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
import skimage
from typing import Optional
from pathlib import Path

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data
from ..data.orientation import assert_same_space
from .utils import largest_island


def create_csf_mask(
    vol: np.ndarray,
    connectivity: Optional[int] = 2,
    use_li: bool = False,
) -> np.ndarray:
    connectivity = connectivity or vol.ndim
    if use_li:
        thresh = skimage.filters.threshold_li(vol)
        binary = vol > thresh
        binary = largest_island(binary, connectivity=connectivity)
    else:
        (hist, bins) = np.histogram(
            vol[(vol > 0) * (vol < np.quantile(vol, 0.999))], bins=512
        )
        thresh = skimage.filters.threshold_yen(hist=(hist, bins))
        binary = vol > thresh
        binary = largest_island(binary, connectivity=connectivity)
    return binary


def csf_mask(
    input: Path,
    connectivity: Optional[int] = 2,
    use_li: bool = False,
    output: Path = None,
) -> MRIData:
    input_vol = load_mri_data(input, dtype=np.single)
    mask = create_csf_mask(input_vol.data, connectivity, use_li)
    assert np.max(mask) > 0, "Masking failed, no voxels in mask"
    mri_data = MRIData(data=mask, affine=input_vol.affine)
    if output is not None:
        save_mri_data(mri_data, output, dtype=np.uint8)
    return mri_data



def create_intracranial_mask(
    csf_mask: MRIData,
    segmentation: MRIData
) -> np.ndarray:
    assert_same_space(csf_mask, segmentation)
    combined_mask = csf_mask.data + segmentation.data.astype(bool)
    background_mask = largest_island(~combined_mask, connectivity=1)
    opened = skimage.morphology.binary_opening(
        background_mask, skimage.morphology.ball(3)
    )
    return ~opened
    #return MRIData(data=~opened, affine=segmentation.affine)


def intracranial_mask(
    csf_mask: Path,
    segmentation: Path,
    output: Optional[Path] = None,
) -> MRIData:
    input_csf_mask = load_mri_data(csf_mask, dtype=bool)
    segmentation_data = load_mri_data(segmentation, dtype=bool)
    mask_data = create_intracranial_mask(input_csf_mask, segmentation_data)
    mri_data = MRIData(data=mask_data, affine=segmentation_data.affine)
    if output is not None:
        save_mri_data(mri_data, output, dtype=np.uint8)
    return mri_data