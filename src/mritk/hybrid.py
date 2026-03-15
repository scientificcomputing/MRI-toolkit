# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import nibabel
import numpy as np
import skimage

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
    logger.debug("Computing hybrid T1 array with threshold %.2f ms.", threshold)
    hybrid = ll_data.copy()
    newmask = mask & (ll_data > threshold) & (mixed_data > threshold)
    hybrid[newmask] = mixed_data[newmask]
    return hybrid


def hybrid_t1map(
    LL_path: Path, mixed_path: Path, csf_mask_path: Path, threshold: float, erode: int = 0, output: Path | None = None
) -> nibabel.nifti1.Nifti1Image:
    """I/O wrapper for merging a Look-Locker and a Mixed T1 map."""
    logger.info(f"Generating hybrid T1 map with threshold {threshold} ms and erosion {erode} voxels.")
    logger.info(f"Loading Look-Locker T1 map from {LL_path}.")
    logger.info(f"Loading Mixed T1 map from {mixed_path}.")
    logger.info(f"Loading CSF mask from {csf_mask_path}.")
    mixed_mri = nibabel.nifti1.load(mixed_path)
    ll_mri = nibabel.nifti1.load(LL_path)

    csf_mask_mri = nibabel.nifti1.load(csf_mask_path)
    csf_mask = csf_mask_mri.get_fdata().astype(bool)

    if erode > 0:
        logger.debug(f"Eroding CSF mask with a ball structuring element of radius {erode}.")
        csf_mask = skimage.morphology.erosion(csf_mask, skimage.morphology.ball(erode))

    hybrid = compute_hybrid_t1_array(ll_mri.get_fdata(), mixed_mri.get_fdata(), csf_mask, threshold)

    hybrid_nii = nibabel.nifti1.Nifti1Image(hybrid, affine=ll_mri.affine, header=ll_mri.header)

    if output is not None:
        logger.info(f"Saving hybrid T1 map to {output}.")
        nibabel.nifti1.save(hybrid_nii, output)
    else:
        logger.info("No output path provided, returning hybrid T1 map as Nifti1Image object.")

    return hybrid_nii


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Add command-line arguments for the hybrid T1 map generation."""
    parser.add_argument("-l", "--input-looklocker", type=Path, required=True, help="Path to the Look-Locker T1 map (NIfTI).")
    parser.add_argument("-m", "--input-mixed", type=Path, required=True, help="Path to the Mixed T1 map (NIfTI).")
    parser.add_argument("-c", "--csf-mask", type=Path, required=True, help="Path to the CSF mask (NIfTI).")
    parser.add_argument("-t", "--threshold", type=float, default=4000.0, help="T1 threshold in ms for substitution.")
    parser.add_argument("-e", "--erode", type=int, default=0, help="Number of voxels to erode the CSF mask.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output path for the hybrid T1 map (NIfTI).")

    if extra_args_cb is not None:
        extra_args_cb(parser)


def dispatch(args):
    """Dispatch function for the hybrid T1 map generation."""

    hybrid_t1map(
        LL_path=args.pop("input_looklocker"),
        mixed_path=args.pop("input_mixed"),
        csf_mask_path=args.pop("csf_mask"),
        threshold=args.pop("threshold"),
        erode=args.pop("erode"),
        output=args.pop("output"),
    )
