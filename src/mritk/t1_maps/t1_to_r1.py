# T1 to R1 module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np
from pathlib import Path
from typing import Union

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data


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
    input_mri: Union[Path, MRIData],
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
