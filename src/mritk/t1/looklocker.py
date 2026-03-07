# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import logging
import numpy as np
import tempfile
import shutil
from functools import partial
from typing import Optional
from pathlib import Path

import tqdm
import skimage

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data
from .utils import mri_facemask, fit_voxel, nan_filter_gaussian, run_dcm2niix

logger = logging.getLogger(__name__)


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


def remove_outliers(data: np.ndarray, mask: np.ndarray, t1_low: float, t1_high: float) -> np.ndarray:
    """
    Applies a mask and removes values outside the physiological T1 range.

    Args:
        data (np.ndarray): 3D array of T1 values.
        mask (np.ndarray): 3D boolean mask of the brain/valid area.
        t1_low (float): Lower physiological limit.
        t1_high (float): Upper physiological limit.

    Returns:
        np.ndarray: A cleaned 3D array with outliers and unmasked regions set to NaN.
    """
    processed = data.copy()
    processed[~mask] = np.nan
    outliers = (processed < t1_low) | (processed > t1_high)
    processed[outliers] = np.nan
    return processed


def create_largest_island_mask(data: np.ndarray, radius: int = 10, erode_dilate_factor: float = 1.3) -> np.ndarray:
    """
    Creates a binary mask isolating the largest contiguous non-NaN region in an array.

    Args:
        data (np.ndarray): The 3D input data containing NaNs and valid values.
        radius (int, optional): The radius for morphological dilation. Defaults to 10.
        erode_dilate_factor (float, optional): Multiplier for the erosion radius
            relative to the dilation radius. Defaults to 1.3.

    Returns:
        np.ndarray: A boolean 3D mask of the largest contiguous island.
    """
    mask = skimage.measure.label(np.isfinite(data))
    regions = skimage.measure.regionprops(mask)
    if not regions:
        return np.zeros_like(data, dtype=bool)

    regions.sort(key=lambda x: x.num_pixels, reverse=True)
    mask = mask == regions[0].label
    try:
        skimage.morphology.remove_small_holes(mask, max_size=10 ** (mask.ndim), connectivity=2, out=mask)
    except TypeError:
        # Older versions of skimage use area_threshold instead of max_size
        skimage.morphology.remove_small_holes(mask, area_threshold=10 ** (mask.ndim), connectivity=2, out=mask)
    skimage.morphology.dilation(mask, skimage.morphology.ball(radius), out=mask)
    skimage.morphology.erosion(mask, skimage.morphology.ball(erode_dilate_factor * radius), out=mask)
    return mask


def compute_looklocker_t1_array(data: np.ndarray, time_s: np.ndarray, t1_roof: float = 10000.0) -> np.ndarray:
    """
    Computes T1 relaxation maps from Look-Locker data using Levenberg-Marquardt fitting.

    Args:
        data (np.ndarray): 4D numpy array (x, y, z, time) of Look-Locker MRI signals.
        time_s (np.ndarray): 1D array of trigger times in seconds.
        t1_roof (float, optional): Maximum allowed T1 value (ms) to cap spurious fits. Defaults to 10000.0.

    Returns:
        np.ndarray: 3D numpy array representing the T1 map in milliseconds. Voxels
        that fail to fit or fall outside the mask are set to NaN.
    """
    assert len(data.shape) >= 4, f"Data should be at least 4-dimensional, got shape {data.shape}"
    mask = mri_facemask(data[..., 0])
    valid_voxels = (np.nanmax(data, axis=-1) > 0) & mask

    data_normalized = np.nan * np.zeros_like(data)
    # Prevent divide by zero warnings dynamically
    max_vals = np.nanmax(data, axis=-1)[valid_voxels, np.newaxis]
    data_normalized[valid_voxels] = data[valid_voxels] / max_vals

    voxel_mask = np.array(np.where(valid_voxels)).T
    d_masked = np.array([data_normalized[i, j, k] for (i, j, k) in voxel_mask])

    with tqdm.tqdm(total=len(d_masked), desc="Fitting Look-Locker Voxels") as pbar:
        voxel_fitter = partial(fit_voxel, time_s, pbar)
        vfunc = np.vectorize(voxel_fitter, signature="(n) -> (3)")
        fitted_coefficients = vfunc(d_masked)

    x2 = fitted_coefficients[:, 1]
    x3 = fitted_coefficients[:, 2]

    i, j, k = voxel_mask.T
    t1map = np.nan * np.zeros_like(data[..., 0])

    # Calculate T1 in ms. Formula: T1 = (x2 / x3)^2 * 1000
    t1map[i, j, k] = (x2 / x3) ** 2 * 1000.0

    return np.minimum(t1map, t1_roof)


def looklocker_t1map_postprocessing(
    T1map: Path,
    T1_low: float,
    T1_high: float,
    radius: int = 10,
    erode_dilate_factor: float = 1.3,
    mask: Optional[np.ndarray] = None,
    output: Path | None = None,
) -> MRIData:
    """
    Performs quality-control and post-processing on a raw Look-Locker T1 map.

    This function cleans up noisy T1 fits by applying a three-step pipeline:
    1. Masking: If no mask is provided, it automatically isolates the brain/head by
       finding the largest contiguous tissue island and applying morphological smoothing.
    2. Outlier Removal: Voxels falling outside the provided physiological bounds
       [T1_low, T1_high] are discarded (set to NaN).
    3. Interpolation: Internal "holes" (NaNs) created by poor fits or outlier
       removal are iteratively filled using a specialized Gaussian filter that
       interpolates from surrounding valid tissue without blurring the edges.

    Args:
        T1map (Path): Path to the raw, unmasked Look-Locker T1 map NIfTI file.
        T1_low (float): Lower physiological limit for T1 values (in ms).
        T1_high (float): Upper physiological limit for T1 values (in ms).
        radius (int, optional): Base radius for morphological dilation when generating
            the automatic mask. Defaults to 10.
        erode_dilate_factor (float, optional): Multiplier for the erosion radius
            relative to the dilation radius to ensure tight mask edges. Defaults to 1.3.
        mask (Optional[np.ndarray], optional): Pre-computed 3D boolean mask. If None,
            one is generated automatically. Defaults to None.
        output (Path | None, optional): Path to save the cleaned T1 map. Defaults to None.

    Returns:
        MRIData: An MRIData object containing the cleaned, masked, and interpolated T1 map.

    Raises:
        RuntimeError: If more than 99% of the voxels are removed during the outlier
            filtering step, indicating a likely unit mismatch (e.g., T1 in seconds instead of ms).
    """
    t1map_mri = load_mri_data(T1map, dtype=np.single)
    t1map_data = t1map_mri.data.copy()

    if mask is None:
        mask = create_largest_island_mask(t1map_data, radius, erode_dilate_factor)

    t1map_data = remove_outliers(t1map_data, mask, T1_low, T1_high)

    if np.isfinite(t1map_data).sum() / t1map_data.size < 0.01:
        raise RuntimeError("After outlier removal, less than 1% of the image is left. Check image units.")

    # Fill internal missing values iteratively using a Gaussian filter
    fill_mask = np.isnan(t1map_data) & mask
    while fill_mask.sum() > 0:
        logger.info(f"Filling in {fill_mask.sum()} voxels within the mask.")
        t1map_data[fill_mask] = nan_filter_gaussian(t1map_data, 1.0)[fill_mask]
        fill_mask = np.isnan(t1map_data) & mask

    processed_T1map = MRIData(t1map_data, t1map_mri.affine)
    if output is not None:
        save_mri_data(processed_T1map, output, dtype=np.single)

    return processed_T1map


def looklocker_t1map(looklocker_input: Path, timestamps: Path, output: Path | None = None) -> MRIData:
    """
    Generates a T1 map from a 4D Look-Locker inversion recovery dataset.

    This function acts as an I/O wrapper. It loads the 4D Look-Locker sequence
    and the corresponding trigger times. It converts the timestamps from milliseconds
    (standard DICOM/text output) to seconds, which is required by the underlying
    exponential fitting math, and triggers the voxel-by-voxel T1 calculation.

    Args:
        looklocker_input (Path): Path to the 4D Look-Locker NIfTI file.
        timestamps (Path): Path to the text file containing the nominal trigger
            delay times (in milliseconds) for each volume in the 4D series.
        output (Path | None, optional): Path to save the resulting T1 map NIfTI file. Defaults to None.

    Returns:
        MRIData: An MRIData object containing the computed 3D T1 map (in milliseconds)
        and the original affine transformation matrix.
    """
    ll_mri = load_mri_data(looklocker_input, dtype=np.single)
    # Convert timestamps from milliseconds to seconds
    time_s = np.loadtxt(timestamps) / 1000.0

    t1map_array = compute_looklocker_t1_array(ll_mri.data, time_s)
    t1map_mri = MRIData(t1map_array.astype(np.single), ll_mri.affine)

    if output is not None:
        save_mri_data(t1map_mri, output, dtype=np.single)

    return t1map_mri


def dicom_to_looklocker(dicomfile: Path, outpath: Path):
    """
    Converts a Look-Locker DICOM file to a standardized NIfTI format.

    Extracts trigger times to a sidecar text file, delegates conversion to dcm2niix,
    and standardizes the output type to single-precision float (intent_code=2001).

    Args:
        dicomfile (Path): Path to the input DICOM file.
        outpath (Path): Desired output path for the converted .nii.gz file.
    """
    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    # Extract and save trigger times
    times = read_dicom_trigger_times(dicomfile)
    np.savetxt(outdir / f"{form}_trigger_times.txt", times)

    with tempfile.TemporaryDirectory(prefix=outpath.stem) as tmpdir:
        tmppath = Path(tmpdir)

        # Delegate heavy lifting to dcm2niix
        run_dcm2niix(dicomfile, tmppath, form, extra_args="-z y --ignore_trigger_times", check=True)

        # Copy metadata sidecar
        shutil.copy(tmppath / f"{form}.json", outpath.with_suffix(".json"))

        # Reload and save to standardize intent codes and precision
        mri = load_mri_data(tmppath / f"{form}.nii.gz", dtype=np.double)
        save_mri_data(mri, outpath.with_suffix(".nii.gz"), dtype=np.single, intent_code=2001)
