# utils.py - Utility functions for T1 mapping and related processing.

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import logging
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy
import skimage

VOLUME_LABELS = [
    "IR-modulus",
    "IR-real",
    "IR-corrected-real",
    "SE-modulus",
    "SE-real",
    "T1map-scanner",
]

logger = logging.getLogger(__name__)


def mri_facemask(vol: np.ndarray, smoothing_level: float = 5.0) -> np.ndarray:
    """
    Generates a binary mask of the head/brain to exclude background air/noise.

    Utilizes Triangle thresholding, morphological hole filling, and Gaussian
    smoothing to create a robust, contiguous mask of the primary subject volume.

    Args:
        vol (np.ndarray): A 3D numpy array of the MRI volume.
        smoothing_level (float, optional): The sigma for the Gaussian blur applied
            to smooth the mask edges. Defaults to 5.

    Returns:
        np.ndarray: A 3D boolean array where True indicates the subject/head.
    """
    thresh = skimage.filters.threshold_triangle(vol)
    binary = vol > thresh
    binary = scipy.ndimage.binary_fill_holes(binary)
    binary = skimage.filters.gaussian(binary, sigma=smoothing_level)
    binary = binary > skimage.filters.threshold_isodata(binary)
    return binary


def nan_filter_gaussian(U: np.ndarray, sigma: float, truncate: float = 4.0, mode: str = "constant") -> np.ndarray:
    """
    Applies a Gaussian filter to an array containing NaNs, smoothly interpolating
    the missing values.

    Standard Gaussian filters pull NaNs into surrounding valid data. This function
    creates a normalized convolution mask (WW) to properly handle edges and missing
    values, allowing NaN "holes" to be cleanly interpolated based only on valid
    surrounding neighbors.

    Args:
        U (np.ndarray): Input array potentially containing NaN values.
        sigma (float): Standard deviation for the Gaussian kernel.
        truncate (float, optional): Truncate the filter at this many standard deviations. Defaults to 4.0.
        mode (str, optional): Mode for handling edges. Defaults to "constant".

    Returns:
        np.ndarray: Filtered array where original NaN values have been interpolated.
    """
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate, mode=mode)

    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate, mode=mode)
    mask = ~((WW == 0) * (VV == 0))
    out = np.nan * np.zeros_like(U)
    out[mask] = VV[mask] / WW[mask]
    return out


def run_dcm2niix(input_path: Path, output_dir: Path, form: str, extra_args: str = "", check: bool = True):
    """
    Utility wrapper to execute the dcm2niix command-line tool securely.

    Args:
        input_path (Path): Path to the input DICOM file/folder.
        output_dir (Path): Path to the target output directory.
        form (str): Output filename format string.
        extra_args (str, optional): Additional command line arguments. Defaults to "".
        check (bool, optional): If True, raises an exception on failure. Defaults to True.

    Raises:
        RuntimeError: If the dcm2niix executable is not found in the system PATH.
        subprocess.CalledProcessError: If the command fails and `check` is True.
    """
    logger.info(f"Running dcm2niix with input: {input_path}, output_dir: {output_dir}, form: {form}, extra_args: '{extra_args}'")
    # 1. Locate the executable securely
    executable = shutil.which("dcm2niix")
    logger.debug(f"Located dcm2niix executable at: {executable}")
    if executable is None:
        raise RuntimeError(
            "The 'dcm2niix' executable was not found. Please ensure it is installed and available in your system PATH."
        )

    # 2. Build the arguments list safely
    args = [executable, "-f", form]

    # Safely parse the extra string arguments into a list
    if extra_args:
        args.extend(shlex.split(extra_args))

    args.extend(["-o", str(output_dir), str(input_path)])

    # Reconstruct the command string purely for logging purposes
    cmd_str = shlex.join(args)
    logger.debug(f"Executing: {cmd_str}")

    try:
        # 3. Execute without shell=True for better security and stability
        logger.debug(f"Attempting to run dcm2niix with arguments: {args}")
        subprocess.run(args, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"dcm2niix execution failed.\nCommand: {cmd_str}\nError: {e.stderr}")
        if check:
            raise
