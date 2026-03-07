# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import logging
import numpy as np
import skimage
import nibabel
from pathlib import Path


logger = logging.getLogger(__name__)


def compute_hybrid_t1_array(ll_data: np.ndarray, mixed_data: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """
    Creates a hybrid T1 array by selectively substituting Look-Locker voxels with Mixed voxels.

    Substitution occurs only if BOTH the Look-Locker AND Mixed T1 values exceed the threshold,
    AND the voxel falls within the provided CSF mask.

    Args:
        ll_data (np.ndarray): 3D numpy array of Look-Locker T1 values.
        mixed_data (np.ndarray): 3D numpy array of Mixed T1 values.
        mask (np.ndarray): 3D boolean mask (typically eroded CSF).
        threshold (float): T1 threshold value (in ms).

    Returns:
        np.ndarray: Hybrid 3D T1 array.
    """
    hybrid = ll_data.copy()
    newmask = mask & (ll_data > threshold) & (mixed_data > threshold)
    hybrid[newmask] = mixed_data[newmask]
    return hybrid


def hybrid_t1map(
    LL_path: Path, mixed_path: Path, csf_mask_path: Path, threshold: float, erode: int = 0, output: Path | None = None
) -> nibabel.nifti1.Nifti1Image:
    """I/O wrapper for merging a Look-Locker and a Mixed T1 map."""
    mixed_mri = nibabel.nifti1.load(mixed_path)
    ll_mri = nibabel.nifti1.load(LL_path)

    csf_mask_mri = nibabel.nifti1.load(csf_mask_path)
    csf_mask = csf_mask_mri.get_fdata().astype(bool)

    if erode > 0:
        csf_mask = skimage.morphology.erosion(csf_mask, skimage.morphology.ball(erode))

    hybrid = compute_hybrid_t1_array(ll_mri.get_fdata(), mixed_mri.get_fdata(), csf_mask, threshold)

    hybrid_nii = nibabel.nifti1.Nifti1Image(hybrid, affine=ll_mri.affine, header=ll_mri.header)
    if output is not None:
        nibabel.nifti1.save(hybrid_nii, output)

    return hybrid_nii
