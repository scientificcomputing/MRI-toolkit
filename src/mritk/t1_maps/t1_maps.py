"""T1 Maps generation module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
import tqdm
from functools import partial
import skimage
from typing import Optional
from pathlib import Path
import nibabel
import json
import scipy

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data
from ..masking.masks import create_csf_mask
from .utils import (
    mri_facemask,
    fit_voxel,
    nan_filter_gaussian,
    T1_lookup_table,
)


def looklocker_t1map(
    looklocker_input: Path,
    timestamps: Path,
    output: Path = None
) -> MRIData:
    LL_mri = load_mri_data(looklocker_input, dtype=np.single)
    D = LL_mri.data
    affine = LL_mri.affine
    t_data = np.loadtxt(timestamps) / 1000

    T1_ROOF = 10000
    assert len(D.shape) >= 4, f"data should be 4-dimensional, got data with shape {D.shape}"
    mask = mri_facemask(D[..., 0])
    valid_voxels = (np.nanmax(D, axis=-1) > 0) * mask

    D_normalized = np.nan * np.zeros_like(D)
    D_normalized[valid_voxels] = D[valid_voxels] / np.nanmax(D, axis=-1)[valid_voxels, np.newaxis]
    voxel_mask = np.array(np.where(valid_voxels)).T
    Dmasked = np.array([D_normalized[i, j, k] for (i, j, k) in voxel_mask])

    with tqdm.tqdm(total=len(Dmasked)) as pbar:
        voxel_fitter = partial(fit_voxel, t_data, pbar)
        vfunc = np.vectorize(voxel_fitter, signature="(n) -> (3)")
        fitted_coefficients = vfunc(Dmasked)

    x1, x2, x3 = (
        fitted_coefficients[:, 0],
        fitted_coefficients[:, 1],
        fitted_coefficients[:, 2],
    )

    I, J, K = voxel_mask.T
    T1map = np.nan * np.zeros_like(D[..., 0])
    T1map[I, J, K] = (x2 / x3) ** 2 * 1000.0  # convert to ms
    T1map = np.minimum(T1map, T1_ROOF)
    T1map_mri = MRIData(T1map.astype(np.single), affine)
    if output is not None:
        save_mri_data(T1map_mri, output, dtype=np.single)

    return T1map_mri


def looklocker_t1map_postprocessing(
    T1map_mri: MRIData,
    T1_low: float,
    T1_high: float,
    radius: int = 10,
    erode_dilate_factor: float = 1.3,
    mask: Optional[np.ndarray] = None,
    output: Path = None
) -> MRIData:
    T1map = T1map_mri.data.copy()
    if mask is None:
        # Create mask for largest island.
        mask = skimage.measure.label(np.isfinite(T1map))
        regions = skimage.measure.regionprops(mask)
        regions.sort(key=lambda x: x.num_pixels, reverse=True)
        mask = mask == regions[0].label
        skimage.morphology.remove_small_holes(mask, 10 ** (mask.ndim), connectivity=2, out=mask)
        skimage.morphology.binary_dilation(mask, skimage.morphology.ball(radius), out=mask)
        skimage.morphology.binary_erosion(mask, skimage.morphology.ball(erode_dilate_factor * radius), out=mask)

    # Remove non-zero artifacts outside of the mask.
    surface_vox = np.isfinite(T1map) * (~mask)
    print(f"Removing {surface_vox.sum()} voxels outside of the head mask")
    T1map[~mask] = np.nan

    # Remove outliers within the mask.
    outliers = np.logical_or(T1map < T1_low, T1_high < T1map)
    print("Removing", outliers.sum(), f"voxels outside the range ({T1_low}, {T1_high}).")
    T1map[outliers] = np.nan
    if np.isfinite(T1map).sum() / T1map.size < 0.01:
        raise RuntimeError("After outlier removal, less than 1% of the image is left. Check image units.")

    # Fill internallly missing values
    fill_mask = np.isnan(T1map) * mask
    while fill_mask.sum() > 0:
        print(f"Filling in {fill_mask.sum()} voxels within the mask.")
        T1map[fill_mask] = nan_filter_gaussian(T1map, 1.0)[fill_mask]
        fill_mask = np.isnan(T1map) * mask

    processed_T1map = MRIData(T1map, T1map_mri.affine)
    if output is not None:
        save_mri_data(processed_T1map, output, dtype=np.single)

    return processed_T1map


def mixed_t1map(
    SE_nii_path: Path,
    IR_nii_path: Path,
    meta_path: Path,
    T1_low: float,
    T1_high: float,
    output: Path = None
) -> nibabel.nifti1.Nifti1Image:
    SE = load_mri_data(SE_nii_path, dtype=np.single)
    IR = load_mri_data(IR_nii_path, dtype=np.single)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    nonzero_mask = SE.data != 0
    F_data = np.nan * np.zeros_like(IR.data)
    F_data[nonzero_mask] = IR.data[nonzero_mask] / SE.data[nonzero_mask]

    TR_se, TI, TE, ETL = meta["TR_SE"], meta["TI"], meta["TE"], meta["ETL"]
    F, T1_grid = T1_lookup_table(TR_se, TI, TE, ETL, T1_low, T1_high)
    interpolator = scipy.interpolate.interp1d(F, T1_grid, kind="nearest", bounds_error=False, fill_value=np.nan)
    T1_volume = interpolator(F_data).astype(np.single)
    nii = nibabel.nifti1.Nifti1Image(T1_volume, IR.affine)
    nii.set_sform(nii.affine, "scanner")
    nii.set_qform(nii.affine, "scanner")

    # T1map_mri = MRIData(T1_volume, nii.affine)
    if output is not None:
        nibabel.nifti1.save(nii, output)
        # save_mri_data(T1map_mri, output, dtype=np.single)

    return nii
    # return T1map_mri


def mixed_t1map_postprocessing(
    SE_nii_path: Path, 
    T1_path: Path, 
    output: Path = None
) -> nibabel.nifti1.Nifti1Image:
    T1map_nii = nibabel.nifti1.load(T1_path)

    SE_mri = load_mri_data(SE_nii_path, np.single)
    mask = create_csf_mask(SE_mri.data, use_li=True)
    mask = skimage.morphology.binary_erosion(mask)

    masked_T1map = T1map_nii.get_fdata(dtype=np.single)
    masked_T1map[~mask] = np.nan
    masked_T1map_nii = nibabel.nifti1.Nifti1Image(masked_T1map, T1map_nii.affine, T1map_nii.header)

    if output is not None:
        nibabel.nifti1.save(masked_T1map_nii, output)

    return masked_T1map_nii


def hybrid_t1map(
    LL_path: Path,
    mixed_path: Path,
    csf_mask_path: Path,
    threshold: float,
    erode: int = 0,
    output: Path = None
) -> nibabel.nifti1.Nifti1Image:
    mixed_mri = nibabel.nifti1.load(mixed_path)
    mixed = mixed_mri.get_fdata()

    ll_mri = nibabel.nifti1.load(LL_path)
    ll = ll_mri.get_fdata()
    csf_mask_mri = nibabel.nifti1.load(csf_mask_path)
    csf_mask = csf_mask_mri.get_fdata().astype(bool)
    if erode > 0:
        csf_mask = skimage.morphology.binary_erosion(csf_mask, skimage.morphology.ball(erode))

    hybrid = ll
    newmask = csf_mask * (ll > threshold) * (mixed > threshold)
    hybrid[newmask] = mixed[newmask]

    hybrid_nii = nibabel.nifti1.Nifti1Image(hybrid, affine=ll_mri.affine, header=ll_mri.header)
    if output is not None:
        nibabel.nifti1.save(hybrid_nii, output)

    return hybrid_nii
