# MRI DICOM to NIfTI conversion - utils

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import numpy as np
import scipy
from pathlib import Path
import skimage
import warnings
from scipy.optimize import OptimizeWarning
import nibabel

from ..data.orientation import data_reorientation, change_of_coordinates_map
from ..data.base import MRIData


VOLUME_LABELS = [
    "IR-modulus",
    "IR-real",
    "IR-corrected-real",
    "SE-modulus",
    "SE-real",
    "T1map-scanner",
]


def read_dicom_trigger_times(dicomfile: Path) -> np.ndarray:
    """
    Extracts unique nominal cardiac trigger delay times from DICOM functional groups.

    Args:
        dicomfile (str): The file path to the DICOM file.

    Returns:
        np.ndarray: A sorted array of unique trigger delay times (in milliseconds)
        extracted from the CardiacSynchronizationSequence.
    """
    import pydicom

    dcm = pydicom.dcmread(dicomfile)
    all_frame_times = [
        f.CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime for f in dcm.PerFrameFunctionalGroupsSequence
    ]
    return np.unique(all_frame_times)


def dicom_standard_affine(frame_fg) -> np.ndarray:
    """
    Generates the DICOM to LPS (Left-Posterior-Superior) affine transformation matrix.

    This maps the voxel coordinate space of a DICOM frame to the physical LPS space
    by utilizing the pixel spacing, slice spacing, and patient orientation cosines.

    Args:
        frame_fg: A DICOM frame functional group sequence object containing
            PixelMeasuresSequence, PlaneOrientationSequence, and PlanePositionSequence.

    Returns:
        np.ndarray: A 4x4 affine transformation matrix mapping from DICOM voxel
        indices to LPS physical coordinates.
    """
    # Get the original data shape
    df = float(frame_fg.PixelMeasuresSequence[0].SpacingBetweenSlices)
    dr, dc = (float(x) for x in frame_fg.PixelMeasuresSequence[0].PixelSpacing)
    plane_orientation = frame_fg.PlaneOrientationSequence[0]
    orientation = np.array(plane_orientation.ImageOrientationPatient)

    # Find orientation of data array relative to LPS-coordinate system.
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    frame_cosine = np.cross(row_cosine, col_cosine)

    # Create DICOM-definition affine map to LPS.
    T_1 = np.array(frame_fg.PlanePositionSequence[0].ImagePositionPatient)

    # Create DICOM-definition affine map to LPS.
    M_dcm = np.zeros((4, 4))
    M_dcm[:3, 0] = row_cosine * dc
    M_dcm[:3, 1] = col_cosine * dr
    M_dcm[:3, 2] = frame_cosine * df
    M_dcm[:3, 3] = T_1
    M_dcm[3, 3] = 1.0

    # Reorder from "natural index order" to DICOM affine map definition order.
    N_order = np.eye(4)[[2, 1, 0, 3]]
    return M_dcm @ N_order


def extract_single_volume(D: np.ndarray, frame_fg) -> MRIData:
    """
    Extracts, scales, and reorients a single DICOM volume into an MRIData object.

    Applies the appropriate RescaleSlope and RescaleIntercept transformations
    to the raw pixel array, and then reorients the resulting data volume from
    the native DICOM LPS space to RAS (Right-Anterior-Superior) space.

    Args:
        D (np.ndarray): The raw 3D pixel array for the volume.
        frame_fg: The corresponding DICOM frame functional group metadata.

    Returns:
        MRIData: A newly constructed MRIData object with scaled pixel values
        and an affine matrix oriented to RAS space.
    """
    # Find scaling values (should potentially be inside scaling loop)
    pixel_value_transform = frame_fg.PixelValueTransformationSequence[0]
    slope = float(pixel_value_transform.RescaleSlope)
    intercept = float(pixel_value_transform.RescaleIntercept)
    private = frame_fg[0x2005, 0x140F][0]
    scale_slope = private[0x2005, 0x100E].value

    # Loop over and scale values.
    volume = np.zeros_like(D, dtype=np.single)
    for idx in range(D.shape[0]):
        volume[idx] = (intercept + slope * D[idx]) / (scale_slope * slope)

    A_dcm = dicom_standard_affine(frame_fg)
    C = change_of_coordinates_map("LPS", "RAS")
    mri = data_reorientation(MRIData(volume, C @ A_dcm))

    return mri


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
        the squared residuals of f(xdata, *popt) - ydata is minimized.
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


def compare_nifti_images(img_path1: Path, img_path2: Path, data_tolerance: float = 0.0) -> bool:
    """
    Compares two NIfTI images for equality of data arrays.

    Provides a robust way to check if two NIfTI files contain identical
    voxel data, accounting for potential NaNs and floating-point inaccuracies.

    Args:
        img_path1 (Path): Path to the first NIfTI file.
        img_path2 (Path): Path to the second NIfTI file.
        data_tolerance (float, optional): Absolute tolerance for floating-point
            comparisons. Use 0.0 for exact mathematical equality. Defaults to 0.0.

    Returns:
        bool: True if images are considered the same, False otherwise.

    Raises:
        AssertionError: If files exist but the data deviates beyond `data_tolerance`.
        FileNotFoundError: If either of the provided file paths does not exist.
    """
    if not img_path1.exists():
        raise FileNotFoundError(f"File not found: {img_path1}")
    if not img_path2.exists():
        raise FileNotFoundError(f"File not found: {img_path2}")

    img1 = nibabel.load(img_path1)
    img2 = nibabel.load(img_path2)

    # 1. Compare Image Data
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    return compare_nifti_arrays(data1, data2, data_tolerance)


def compare_nifti_arrays(arr1: np.ndarray, arr2: np.ndarray, data_tolerance: float = 0.0) -> bool:
    """
    Compares two NIfTI data arrays for equality, accounting for NaNs and tolerance.

    Args:
        arr1 (np.ndarray): The first data array to compare.
        arr2 (np.ndarray): The second data array to compare.
        data_tolerance (float, optional): Absolute tolerance for floating-point
            comparisons. Use 0.0 for exact mathematical equality. Defaults to 0.0.

    Returns:
        bool: True if arrays are considered the same, False otherwise.
    """
    # Convert NaN to zero (can have NaNs in concentration maps)
    arr1 = np.nan_to_num(arr1, nan=0.0)
    arr2 = np.nan_to_num(arr2, nan=0.0)

    if data_tolerance > 0:
        return np.allclose(arr1, arr2, atol=data_tolerance)
    else:
        return np.array_equal(arr1, arr2)
