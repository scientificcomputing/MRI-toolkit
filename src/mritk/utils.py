# utils.py - Utility functions for T1 mapping and related processing.

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

from pathlib import Path
import subprocess
import shutil
import shlex
import numpy as np
import scipy
import skimage
import warnings
import logging
from scipy.optimize import OptimizeWarning


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


def voxel_fit_function(t: np.ndarray, x1: float, x2: float, x3: float) -> np.ndarray:
    """
    Theoretical Look-Locker T1 recovery curve model.

    Evaluates the function: f(t) = | x1 * (1 - (1 + x2^2) * exp(-x3^2 * t)) |

    Args:
        t (np.ndarray): Time array in seconds.
        x1 (float): Amplitude scaling factor (equivalent to A).
        x2 (float): Inversion efficiency term (used to ensure (1+x2^2) > 1).
        x3 (float): Relaxation rate, defined as 1 / sqrt(T1*).

    Returns:
        np.ndarray: The theoretical signal magnitude at times `t`.
    """
    return np.abs(x1 * (1.0 - (1 + x2**2) * np.exp(-(x3**2) * t)))


@np.errstate(divide="raise", invalid="raise", over="raise")
def curve_fit_wrapper(f, t: np.ndarray, y: np.ndarray, p0: np.ndarray):
    """
    A strict wrapper around scipy.optimize.curve_fit.

    Temporarily converts numpy warnings (like division by zero) and
    scipy's OptimizeWarning into hard errors. This allows the calling
    function to gracefully catch and handle poorly-fitting voxels
    (e.g., by assigning them NaN) rather than silently returning bad fits.

    Args:
        f (callable): The model function, e.g., voxel_fit_function.
        t (np.ndarray): The independent variable (time).
        y (np.ndarray): The dependent variable (signal).
        p0 (np.ndarray): Initial guesses for the parameters.

    Returns:
        np.ndarray: Optimal values for the parameters so that the sum of
        the squared residuals of :code:`f(xdata, *popt) - ydata` is minimized.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        popt, _ = scipy.optimize.curve_fit(f, xdata=t, ydata=y, p0=p0, maxfev=1000)
    return popt


def fit_voxel(time_s: np.ndarray, pbar, m: np.ndarray) -> np.ndarray:
    """
    Fits the Look-Locker relaxation curve for a single voxel's time series.

    Provides initial parameter guesses based on the location of the signal minimum
    and attempts to fit the voxel_fit_function using Levenberg-Marquardt optimization.
    Returns NaNs if the optimization fails or hits evaluation limits.

    Args:
        time_s (np.ndarray): 1D array of trigger times in seconds.
        pbar: A tqdm progress bar instance (or None) to update incrementally.
        m (np.ndarray): 1D array of signal magnitudes over time for the voxel.

    Returns:
        np.ndarray: A 3-element array containing the fitted parameters `[x1, x2, x3]`.
        If the fit fails, returns an array of NaNs.
    """
    if pbar is not None:
        pbar.update(1)
    x1 = 1.0
    x2 = np.sqrt(1.25)
    T1 = time_s[np.argmin(m)] / np.log(1 + x2**2)
    x3 = np.sqrt(1 / T1)
    p0 = np.array((x1, x2, x3))
    if not np.all(np.isfinite(m)):
        return np.nan * np.zeros_like(p0)
    try:
        popt = curve_fit_wrapper(voxel_fit_function, time_s, m, p0)
    except (OptimizeWarning, FloatingPointError):
        return np.nan * np.zeros_like(p0)
    except RuntimeError as e:
        if "maxfev" in str(e):
            return np.nan * np.zeros_like(p0)
        raise e
    return popt


def nan_filter_gaussian(U: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
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

    Returns:
        np.ndarray: Filtered array where original NaN values have been interpolated.
    """
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)
    mask = ~((WW == 0) * (VV == 0))
    out = np.nan * np.zeros_like(U)
    out[mask] = VV[mask] / WW[mask]
    return out


def estimate_se_free_relaxation_time(TRse: float, TE: float, ETL: int) -> float:
    """
    Computes the estimated free relaxation time following a Spin Echo image.

    Corrects the standard Repetition Time (TR) by accounting for the Effective
    Echo Time (TE), the Echo Train Length (ETL), and an adjustment for 20
    dummy preparation echoes.

    Args:
        TRse (float): Repetition time of the spin echo sequence (in ms).
        TE (float): Effective echo time (in ms).
        ETL (int): Echo train length.

    Returns:
        float: The corrected free relaxation time `TRfree`.
    """
    return TRse - TE * (1 + 0.5 * (ETL - 1) / (0.5 * (ETL + 1) + 20))


def T1_lookup_table(TRse: float, TI: float, TE: float, ETL: int, T1_low: float, T1_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a Fraction/T1 lookup table for mixed T1 mapping interpolations.

    Calculates the theoretical ratio of the Inversion Recovery signal (Sir) to
    the Spin Echo signal (Sse) over a highly discretized grid of physiological
    T1 relaxation times.

    Args:
        TRse (float): Spin-echo repetition time (in ms).
        TI (float): Inversion time (in ms).
        TE (float): Effective echo time (in ms).
        ETL (int): Echo train length.
        T1_low (float): Lower bound of the T1 grid (in ms).
        T1_hi (float): Upper bound of the T1 grid (in ms).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - fractionCurve (np.ndarray): The theoretical Sir/Sse signal ratios.
            - T1_grid (np.ndarray): The corresponding T1 values (in ms).
    """
    TRfree = estimate_se_free_relaxation_time(TRse, TE, ETL)
    T1_grid = np.arange(int(T1_low), int(T1_hi + 1))
    Sse = 1 - np.exp(-TRfree / T1_grid)
    Sir = 1 - (1 + Sse) * np.exp(-TI / T1_grid)
    fractionCurve = Sir / Sse
    return fractionCurve, T1_grid


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
