# T1 to R1 module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .data import MRIData

logger = logging.getLogger(__name__)


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
    logger.debug(f"Computing R1 array with scale={scale}, t1_low={t1_low}, t1_high={t1_high}")
    valid_t1 = (t1_low <= t1_data) & (t1_data <= t1_high)
    r1_data = np.nan * np.zeros_like(t1_data)

    # Calculate R1 only for valid voxels to avoid division by zero or extreme outliers
    r1_data[valid_t1] = scale / t1_data[valid_t1]

    return r1_data


def convert_t1_to_r1(
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
    logger.debug(f"Converted T1 map to R1 map with shape {r1_data.shape}")
    return MRIData(data=r1_data, affine=T1map_mri.affine)


def t1_to_r1(
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
    logger.info(f"Converting T1 map to R1 map with input: {input_mri}, output: {output}")
    if isinstance(input_mri, Path):
        mri_t1 = MRIData.from_file(input_mri, dtype=np.single)
    elif isinstance(input_mri, MRIData):
        mri_t1 = input_mri
    else:
        raise ValueError(f"Input should be a Path or MRIData, got {type(input_mri)}")

    mri_r1 = convert_t1_to_r1(mri_t1, scale, t1_low, t1_high)

    if output is not None:
        logger.info(f"Saving R1 map to {output}")
        mri_r1.save(output, dtype=np.single)
    else:
        logger.info("No output path provided, returning R1 map as MRIData object")

    return mri_r1


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Add command-line arguments for the T1 to R1 conversion."""
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input T1 map (NIfTI).")
    parser.add_argument("-o", "--output", type=Path, help="Path to save the output R1 map (NIfTI).")
    parser.add_argument("--scale", type=float, default=1000.0, help="Scaling factor for R1 calculation.")
    parser.add_argument("--t1-low", type=float, default=1.0, help="Lower bound for valid T1 values.")
    parser.add_argument("--t1-high", type=float, default=float("inf"), help="Upper bound for valid T1 values.")
    if extra_args_cb is not None:
        extra_args_cb(parser)


def dispatch(args: dict):
    """Dispatch function for the T1 to R1 conversion."""
    t1_to_r1(
        input_mri=args.pop("input"),
        output=args.pop("output"),
        scale=args.pop("scale"),
        t1_low=args.pop("t1_low"),
        t1_high=args.pop("t1_high"),
    )
