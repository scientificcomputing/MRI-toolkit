"""MRI DICOM to NIfTI conversion - utils

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
import numpy as np
import nibabel
import logging
import scipy as sp
import skimage
import warnings
from scipy.optimize import OptimizeWarning

from ..data.orientation import data_reorientation, change_of_coordinates_map
from ..data.base import MRIData

logger = logging.getLogger(__name__)

VOLUME_LABELS = [
    "IR-modulus",
    "IR-real",
    "IR-corrected-real",
    "SE-modulus",
    "SE-real",
    "T1map-scanner",
]


def read_dicom_trigger_times(dicomfile):
    import pydicom

    dcm = pydicom.dcmread(dicomfile)
    all_frame_times = [
        f.CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime
        for f in dcm.PerFrameFunctionalGroupsSequence
    ]
    return np.unique(all_frame_times)


def dicom_standard_affine(
    frame_fg,
) -> np.ndarray:
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


def extract_single_volume(
    D: np.ndarray,
    frame_fg,
) -> MRIData:
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


def mri_facemask(vol: np.ndarray, smoothing_level=5):
    thresh = skimage.filters.threshold_triangle(vol)
    binary = vol > thresh
    binary = sp.ndimage.binary_fill_holes(binary)
    binary = skimage.filters.gaussian(binary, sigma=smoothing_level)
    binary = binary > skimage.filters.threshold_isodata(binary)
    return binary


def curve_fit_wrapper(f, t, y, p0):
    """Raises error instead of catching numpy warnings, such that
    these cases may be treated."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        popt, _ = sp.optimize.curve_fit(f, xdata=t, ydata=y, p0=p0, maxfev=1000)
    return popt
 

def fit_voxel(time_s: np.ndarray, pbar, m: np.ndarray) -> np.ndarray:
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
        popt = curve_fit_wrapper(f, time_s, m, p0)
    except (OptimizeWarning, FloatingPointError):
        return np.nan * np.zeros_like(p0)
    except RuntimeError as e:
        if "maxfev" in str(e):
            return np.nan * np.zeros_like(p0)
        raise e
    return popt


def nan_filter_gaussian(
    U: np.ndarray, sigma: float, truncate: float = 4.0
) -> np.ndarray:
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = sp.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = sp.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)
    mask = ~((WW == 0) * (VV == 0))
    out = np.nan * np.zeros_like(U)
    out[mask] = VV[mask] / WW[mask]
    return out


def estimate_se_free_relaxation_time(TRse, TE, ETL):
    """Compute free relaxation time following spin echo image from effective echo
    time TE and echo train length ETL, corrected for 20 dummy echoes."""
    return TRse - TE * (1 + 0.5 * (ETL - 1) / (0.5 * (ETL + 1) + 20))


def T1_lookup_table(
    TRse: float, TI: float, TE: float, ETL: int, T1_low: float, T1_hi: float
) -> tuple[np.ndarray, np.ndarray]:
    TRfree = estimate_se_free_relaxation_time(TRse, TE, ETL)
    T1_grid = np.arange(int(T1_low), int(T1_hi + 1))
    Sse = 1 - np.exp(-TRfree / T1_grid)
    Sir = 1 - (1 + Sse) * np.exp(-TI / T1_grid)
    fractionCurve = Sir / Sse
    return fractionCurve, T1_grid