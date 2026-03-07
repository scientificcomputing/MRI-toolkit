# T1 to R1 module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np
from pathlib import Path

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data
from ..data.orientation import assert_same_space


def compute_r1_array(
    t1_data: np.ndarray, scale: float = 1000.0, t1_low: float = 1.0, t1_high: float = float("inf")
) -> np.ndarray:
    """
    Pure numpy function converting a T1 relaxation time array to an R1 relaxation rate array.

    The relationship is R1 = scale / T1. Values outside the [t1_low, t1_high]
    range are set to NaN to filter out noise and non-physiological data.

    Args:
        t1_data (np.ndarray): The input array containing T1 relaxation times.
        scale (float, optional): Scaling factor, typically 1000 to convert from ms to s^-1. Defaults to 1000.
        t1_low (float, optional): Lower bound for valid T1 values. Defaults to 1.
        t1_high (float, optional): Upper bound for valid T1 values. Defaults to infinity.

    Returns:
        np.ndarray: An array of R1 relaxation rates. Invalid/out-of-bound voxels are set to NaN.
    """
    valid_t1 = (t1_low <= t1_data) & (t1_data <= t1_high)
    r1_data = np.nan * np.zeros_like(t1_data)

    # Calculate R1 only for valid voxels to avoid division by zero or extreme outliers
    r1_data[valid_t1] = scale / t1_data[valid_t1]

    return r1_data


def convert_T1_to_R1(
    T1map_mri: MRIData,
    scale: float = 1000.0,
    t1_low: float = 1.0,
    t1_high: float = float("inf"),
) -> MRIData:
    """
    Converts a T1 map MRIData object into an R1 map MRIData object.

    Args:
        T1map_mri (MRIData): The input MRIData object representing the T1 map.
        scale (float, optional): Scaling factor. Defaults to 1000.
        t1_low (float, optional): Lower bound for valid T1 values. Defaults to 1.
        t1_high (float, optional): Upper bound for valid T1 values. Defaults to float('inf').

    Returns:
        MRIData: A new MRIData object containing the R1 map array and the original affine matrix.
    """
    r1_data = compute_r1_array(T1map_mri.data, scale, t1_low, t1_high)
    return MRIData(data=r1_data, affine=T1map_mri.affine)


def T1_to_R1(
    input_mri: Path | MRIData,
    output: Path | None = None,
    scale: float = 1000.0,
    t1_low: float = 1.0,
    t1_high: float = float("inf"),
) -> MRIData:
    """
    High-level wrapper to convert a T1 map to an R1 map, handling file I/O operations.

    Args:
        input_mri (Union[Path, MRIData]): A Path to a T1 NIfTI file or an already loaded MRIData object.
        output (Path | None, optional): Path to save the resulting R1 map to disk. Defaults to None.
        scale (float, optional): Scaling factor (e.g., 1000 for ms -> s^-1). Defaults to 1000.
        t1_low (float, optional): Lower bound for valid T1 values. Defaults to 1.
        t1_high (float, optional): Upper bound for valid T1 values. Defaults to float('inf').

    Returns:
        MRIData: The computed R1 map as an MRIData object.

    Raises:
        ValueError: If input_mri is neither a Path nor an MRIData object.
    """
    if isinstance(input_mri, Path):
        T1map_mri = load_mri_data(input_mri, dtype=np.single)
    elif isinstance(input_mri, MRIData):
        T1map_mri = input_mri
    else:
        raise ValueError(f"Input should be a Path or MRIData, got {type(input_mri)}")

    R1map_mri = convert_T1_to_R1(T1map_mri, scale, t1_low, t1_high)

    if output is not None:
        save_mri_data(R1map_mri, output, dtype=np.single)

    return R1map_mri


def concentration_from_T1(t1: np.ndarray, t1_0: np.ndarray, r1: float) -> np.ndarray:
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


def concentration_from_R1(r1_map: np.ndarray, r1_0_map: np.ndarray, r1: float) -> np.ndarray:
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


def compute_concentration_array(
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
    concentrations[valid_mask] = concentration_from_T1(t1=t1_data[valid_mask], t1_0=t10_data[valid_mask], r1=r1)

    return concentrations


def concentration(
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

    concentrations_array = compute_concentration_array(t1_data=t1_mri.data, t10_data=t10_mri.data, r1=r1, mask=mask_data)

    mri_data = MRIData(data=concentrations_array, affine=t10_mri.affine)

    if output_path is not None:
        save_mri_data(mri_data, output_path, dtype=np.single)

    return mri_data
