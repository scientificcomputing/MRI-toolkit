#  Concentration module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np
from pathlib import Path

from .data.base import MRIData
from .data.io import load_mri_data, save_mri_data
from .data.orientation import assert_same_space


def concentration_from_T1_expr(t1: np.ndarray, t1_0: np.ndarray, r1: float) -> np.ndarray:
    """
    Computes tracer concentration from T1 relaxation times.

    Formula: C = (1 / r1) * ((1 / T1) - (1 / T1_0))

    Args:
        t1 (np.ndarray): Array of post-contrast T1 relaxation times.
        t1_0 (np.ndarray): Array of pre-contrast (baseline) T1 relaxation times.
        r1 (float): Relaxivity of the contrast agent.

    Returns:
        np.ndarray: Computed concentration array.
    """
    return (1.0 / r1) * ((1.0 / t1) - (1.0 / t1_0))


def concentration_from_R1_expr(r1_map: np.ndarray, r1_0_map: np.ndarray, r1: float) -> np.ndarray:
    """
    Computes tracer concentration from R1 relaxation rates.

    Formula: C = (1 / r1) * (R1 - R1_0)

    Args:
        r1_map (np.ndarray): Array of post-contrast R1 relaxation rates.
        r1_0_map (np.ndarray): Array of pre-contrast (baseline) R1 relaxation rates.
        r1 (float): Relaxivity of the contrast agent.

    Returns:
        np.ndarray: Computed concentration array.
    """
    return (1.0 / r1) * (r1_map - r1_0_map)


def compute_concentration_from_T1_array(
    t1_data: np.ndarray, t10_data: np.ndarray, r1: float, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Computes the concentration map array, handling masking and avoiding division by zero.

    Args:
        t1_data (np.ndarray): 3D numpy array of post-contrast T1 values.
        t10_data (np.ndarray): 3D numpy array of pre-contrast T1 values.
        r1 (float): Relaxivity of the contrast agent.
        mask (Optional[np.ndarray], optional): Boolean mask restricting the computation area.
            Defaults to None.

    Returns:
        np.ndarray: A 3D array of computed concentrations. Invalid voxels (unmasked or
        where T1 <= 1e-10) are set to NaN.
    """
    # Create a validity mask: T1 values must be > 1e-10 to safely invert without overflow
    valid_mask = (t1_data > 1e-10) & (t10_data > 1e-10)

    if mask is not None:
        valid_mask &= mask.astype(bool)

    concentrations = np.full_like(t10_data, np.nan, dtype=np.single)

    # Compute concentration strictly on valid voxels
    concentrations[valid_mask] = concentration_from_T1_expr(t1=t1_data[valid_mask], t1_0=t10_data[valid_mask], r1=r1)

    return concentrations


def concentration_from_T1(
    input_path: Path,
    reference_path: Path,
    output_path: Path | None = None,
    r1: float = 0.0045,
    mask_path: Path | None = None,
) -> MRIData:
    """
    I/O wrapper to generate a contrast agent concentration map from NIfTI T1 maps.

    Loads the post-contrast and baseline T1 maps, ensures they occupy the same
    physical space, computes the concentration map, and optionally saves it to disk.

    Args:
        input_path (Path): Path to the post-contrast T1 map NIfTI file.
        reference_path (Path): Path to the baseline (pre-contrast) T1 map NIfTI file.
        output_path (Path | None, optional): Path to save the resulting concentration map. Defaults to None.
        r1 (float, optional): Contrast agent relaxivity. Defaults to 0.0045.
        mask_path (Path | None, optional): Path to a boolean mask NIfTI file. Defaults to None.

    Returns:
        MRIData: An MRIData object containing the concentration array and the affine matrix.
    """
    t1_mri = load_mri_data(input_path, dtype=np.single)
    t10_mri = load_mri_data(reference_path, dtype=np.single)
    assert_same_space(t1_mri, t10_mri)

    mask_data = None
    if mask_path is not None:
        mask_mri = load_mri_data(mask_path, dtype=bool)
        assert_same_space(mask_mri, t10_mri)
        mask_data = mask_mri.data

    concentrations_array = compute_concentration_from_T1_array(t1_data=t1_mri.data, t10_data=t10_mri.data, r1=r1, mask=mask_data)

    mri_data = MRIData(data=concentrations_array, affine=t10_mri.affine)

    if output_path is not None:
        save_mri_data(mri_data, output_path, dtype=np.single)

    return mri_data


def compute_concentration_from_R1_array(
    r1_data: np.ndarray, r10_data: np.ndarray, r1: float, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Computes the concentration map array from R1 maps, handling masking.

    Unlike T1 maps, R1 calculations do not suffer from division-by-zero
    errors, but we still ensure we only operate on finite values and within
    the provided mask.

    Args:
        r1_data (np.ndarray): 3D numpy array of post-contrast R1 values.
        r10_data (np.ndarray): 3D numpy array of pre-contrast R1 values.
        r1 (float): Relaxivity of the contrast agent.
        mask (np.ndarray | None, optional): Boolean mask restricting the computation area.
            Defaults to None.

    Returns:
        np.ndarray: A 3D array of computed concentrations. Invalid voxels (unmasked
        or where R1 is not finite) are set to NaN.
    """
    # Create a validity mask: limit to finite floating point numbers
    valid_mask = np.isfinite(r1_data) & np.isfinite(r10_data)

    if mask is not None:
        valid_mask &= mask.astype(bool)

    concentrations = np.full_like(r10_data, np.nan, dtype=np.single)

    # Compute concentration strictly on valid voxels
    concentrations[valid_mask] = concentration_from_R1_expr(r1_map=r1_data[valid_mask], r1_0_map=r10_data[valid_mask], r1=r1)

    return concentrations


def concentration_from_R1(
    input_path: Path,
    reference_path: Path,
    output_path: Path | None = None,
    r1: float = 0.0045,
    mask_path: Path | None = None,
) -> MRIData:
    """
    I/O wrapper to generate a contrast agent concentration map from NIfTI R1 maps.

    Loads the post-contrast and baseline R1 maps, ensures they occupy the same
    physical space, computes the concentration map, and optionally saves it to disk.

    Args:
        input_path (Path): Path to the post-contrast R1 map NIfTI file.
        reference_path (Path): Path to the baseline (pre-contrast) R1 map NIfTI file.
        output_path (Path | None, optional): Path to save the resulting concentration map. Defaults to None.
        r1 (float, optional): Contrast agent relaxivity. Defaults to 0.0045.
        mask_path (Path | None, optional): Path to a boolean mask NIfTI file. Defaults to None.

    Returns:
        MRIData: An MRIData object containing the concentration array and the affine matrix.
    """
    r1_mri = load_mri_data(input_path, dtype=np.single)
    r10_mri = load_mri_data(reference_path, dtype=np.single)
    assert_same_space(r1_mri, r10_mri)

    mask_data = None
    if mask_path is not None:
        mask_mri = load_mri_data(mask_path, dtype=bool)
        assert_same_space(mask_mri, r10_mri)
        mask_data = mask_mri.data

    concentrations_array = compute_concentration_from_R1_array(r1_data=r1_mri.data, r10_data=r10_mri.data, r1=r1, mask=mask_data)

    mri_data = MRIData(data=concentrations_array, affine=r10_mri.affine)

    if output_path is not None:
        save_mri_data(mri_data, output_path, dtype=np.single)

    return mri_data
