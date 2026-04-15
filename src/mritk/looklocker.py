# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import skimage
import tqdm

from .data import MRIData
from .utils import fit_voxel, mri_facemask, nan_filter_gaussian, run_dcm2niix

logger = logging.getLogger(__name__)


@dataclass
class LookLocker:
    """
    A class encapsulating Look-Locker T1 map generation and post-processing.

    Args:
        mri (MRIData): An MRIData object containing the 4D Look-Locker sequence.
        times (np.ndarray): A 1D array of trigger delay times in seconds corresponding to each volume in the 4D sequence.

    """

    mri: MRIData
    times: np.ndarray

    def t1_map(self) -> "LookLockerT1":
        """
        Computes the T1 map from the Look-Locker data using the provided trigger times.

        Returns:
            LookLockerT1: A LookLockerT1 object containing the computed 3D T1 map (in milliseconds)
            and the original affine transformation matrix.
        """

        logger.info("Generating T1 map from Look-Locker data")
        t1map_array = compute_looklocker_t1_array(self.mri.data, self.times)
        mri_data = MRIData(t1map_array.astype(np.single), self.mri.affine)
        return LookLockerT1(mri=mri_data)

    @classmethod
    def from_file(cls, looklocker_input: Path, timestamps: Path):
        logger.info(f"Loading Look-Locker data from {looklocker_input} and trigger times from {timestamps}.")
        ll_mri = MRIData.from_file(looklocker_input, dtype=np.single)
        time_s = np.loadtxt(timestamps) / 1000.0
        logger.debug(f"Loaded trigger times: {time_s}.")
        return cls(mri=ll_mri, times=time_s)


@dataclass
class LookLockerT1:
    """A class representing a Look-Locker T1 map with post-processing capabilities.

    Args:
        mri (MRIData): An MRIData object containing the raw T1 map data and affine transformation.
    """

    mri: MRIData

    @classmethod
    def from_file(cls, t1map_path: Path) -> "LookLockerT1":
        """Loads a Look-Locker T1 map from a NIfTI file.

        Args:
            t1map_path (Path): The file path to the Look-Locker T1 map
            NIfTI file.
        Returns:
            LookLockerT1: An instance of the LookLockerT1 class containing the loaded
            T1 map data and affine transformation.
        """

        logger.info(f"Loading Look-Locker T1 map from {t1map_path}.")
        mri = MRIData.from_file(t1map_path, dtype=np.single)
        return cls(mri=mri)

    def postprocess(
        self,
        T1_low: float = 100,
        T1_high: float = 6000,
        radius: int = 10,
        erode_dilate_factor: float = 1.3,
        mask: np.ndarray | None = None,
    ) -> MRIData:
        """Performs quality-control and post-processing on a raw Look-Locker T1 map.

        Args:
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

        Notes:
            This function cleans up noisy T1 fits by applying a three-step pipeline:
            1. Masking: If no mask is provided, it automatically isolates the brain/head by
            finding the largest contiguous tissue island and applying morphological smoothing.
            2. Outlier Removal: Voxels falling outside the provided physiological bounds
            [T1_low, T1_high] are discarded (set to NaN).
            3. Interpolation: Internal "holes" (NaNs) created by poor fits or outlier
            removal are iteratively filled using a specialized Gaussian filter that
            interpolates from surrounding valid tissue without blurring the edges.
        """
        logger.info(f"Post-processing Look-Locker T1 map with T1 range [{T1_low}, {T1_high}] ms.")
        t1map_data = self.mri.data.copy()

        if mask is None:
            logger.debug("No mask provided, generating automatic mask based on the largest contiguous tissue island.")
            mask = create_largest_island_mask(t1map_data, radius, erode_dilate_factor)
        else:
            logger.debug("Using provided mask for post-processing.")

        t1map_data = remove_outliers(t1map_data, mask, T1_low, T1_high)

        if np.isfinite(t1map_data).sum() / t1map_data.size < 0.01:
            raise RuntimeError("After outlier removal, less than 1% of the image is left. Check image units.")

        # Fill internal missing values iteratively using a Gaussian filter
        fill_mask = np.isnan(t1map_data) & mask
        logger.debug(f"Initial fill mask has {fill_mask.sum()} voxels.")
        while fill_mask.sum() > 0:
            logger.info(f"Filling in {fill_mask.sum()} voxels within the mask.")
            t1map_data[fill_mask] = nan_filter_gaussian(t1map_data, 1.0)[fill_mask]
            fill_mask = np.isnan(t1map_data) & mask

        return MRIData(t1map_data, self.mri.affine)


def read_dicom_trigger_times(dicomfile: Path) -> np.ndarray:
    """
    Extracts unique nominal cardiac trigger delay times from DICOM functional groups.

    Args:
        dicomfile (str): The file path to the DICOM file.

    Returns:
        np.ndarray: A sorted array of unique trigger delay times (in milliseconds)
        extracted from the CardiacSynchronizationSequence.
    """
    logger.info(f"Reading DICOM trigger times from {dicomfile}.")
    import pydicom

    dcm = pydicom.dcmread(dicomfile)
    all_frame_times = [
        f.CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime for f in dcm.PerFrameFunctionalGroupsSequence
    ]

    return np.unique(all_frame_times)


def remove_outliers(data: np.ndarray, mask: np.ndarray | None = None, t1_low: float = 50, t1_high: float = 5000) -> np.ndarray:
    """
    Applies a mask and removes values outside the physiological T1 range.

    Args:
        data (np.ndarray): 3D array of T1 values.
        mask (np.ndarray | None): 3D boolean mask of the brain/valid area.
        t1_low (float): Lower physiological limit.
        t1_high (float): Upper physiological limit.

    Returns:
        np.ndarray: A cleaned 3D array with outliers and unmasked regions set to NaN.
    """
    logger.info("Removing outliers from T1 map with physiological range [%f, %f].", t1_low, t1_high)
    processed = data.copy()
    if mask is not None:
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
    logger.info("Creating largest island mask with dilation radius %d and erosion factor %.2f.", radius, erode_dilate_factor)
    mask = skimage.measure.label(np.isfinite(data))
    logger.debug("Region properties calculated for %d labeled regions.", mask.max())
    regions = skimage.measure.regionprops(mask)
    if not regions:
        return np.zeros_like(data, dtype=bool)

    logger.debug("Sorting regions by size to identify the largest contiguous island.")
    regions.sort(key=lambda x: x.num_pixels, reverse=True)
    mask = mask == regions[0].label
    try:
        logger.debug("Removing small holes with max_size %d.", 10 ** (mask.ndim))
        skimage.morphology.remove_small_holes(mask, max_size=10 ** (mask.ndim), connectivity=2, out=mask)
    except TypeError:
        # Older versions of skimage use area_threshold instead of max_size
        skimage.morphology.remove_small_holes(mask, area_threshold=10 ** (mask.ndim), connectivity=2, out=mask)
    logger.debug("Applying morphological dilation with radius %d.", radius)
    skimage.morphology.dilation(mask, skimage.morphology.ball(radius), out=mask)
    logger.debug("Applying morphological erosion with radius %d.", erode_dilate_factor * radius)
    skimage.morphology.erosion(mask, skimage.morphology.ball(erode_dilate_factor * radius), out=mask)
    logger.debug(f"Generated final mask with shape {mask.shape} and {mask.sum()} valid voxels.")
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
    logger.info("Computing Look-Locker T1 map from 4D data with shape %s and trigger times %s.", data.shape, time_s)
    assert len(data.shape) >= 4, f"Data should be at least 4-dimensional, got shape {data.shape}"
    mask = mri_facemask(data[..., 0])
    logger.debug(f"Generated face mask with shape {mask.shape} and {mask.sum()} valid voxels.")
    valid_voxels = (np.nanmax(data, axis=-1) > 0) & mask
    logger.debug(f"Identified {valid_voxels.sum()} valid voxels for fitting after applying mask and signal threshold.")

    data_normalized = np.nan * np.zeros_like(data)
    # Prevent divide by zero warnings dynamically
    max_vals = np.nanmax(data, axis=-1)[valid_voxels, np.newaxis]
    data_normalized[valid_voxels] = data[valid_voxels] / max_vals

    voxel_mask = np.array(np.where(valid_voxels)).T
    d_masked = np.array([data_normalized[i, j, k] for (i, j, k) in voxel_mask])

    logger.debug(f"Starting fitting for {len(d_masked)} voxels.")
    with tqdm.tqdm(total=len(d_masked), desc="Fitting Look-Locker Voxels") as pbar:
        voxel_fitter = partial(fit_voxel, time_s=time_s, pbar=pbar)
        vfunc = np.vectorize(voxel_fitter, signature="(n) -> (3)")
        fitted_coefficients = vfunc(m=d_masked)

    x2 = fitted_coefficients[:, 1]
    x3 = fitted_coefficients[:, 2]

    i, j, k = voxel_mask.T
    t1map = np.nan * np.zeros_like(data[..., 0])

    # Calculate T1 in ms. Formula: T1 = (x2 / x3)^2 * 1000
    t1map[i, j, k] = (x2 / x3) ** 2 * 1000.0

    return np.minimum(t1map, t1_roof)


def dicom_to_looklocker(dicomfile: Path, outpath: Path):
    """
    Converts a Look-Locker DICOM file to a standardized NIfTI format.

    Extracts trigger times to a sidecar text file, delegates conversion to dcm2niix,
    and standardizes the output type to single-precision float (intent_code=2001).

    Args:
        dicomfile (Path): Path to the input DICOM file.
        outpath (Path): Desired output path for the converted .nii.gz file.
    """
    logger.info(f"Converting Look-Locker DICOM {dicomfile} to NIfTI format at {outpath}")
    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    # Extract and save trigger times
    times = read_dicom_trigger_times(dicomfile)
    trigger_file = outdir / f"{form}_trigger_times.txt"
    logger.debug(f"Extracted trigger times: {times}. Saving to {trigger_file}")
    np.savetxt(trigger_file, times)

    with tempfile.TemporaryDirectory(prefix=outpath.stem) as tmpdir:
        logger.debug(f"Created temporary directory {tmpdir} for intermediate dcm2niix output.")
        tmppath = Path(tmpdir)

        # Delegate heavy lifting to dcm2niix
        run_dcm2niix(dicomfile, tmppath, form, extra_args="-z y --ignore_trigger_times", check=True)

        # Copy metadata sidecar
        shutil.copy(tmppath / f"{form}.json", outpath.with_suffix(".json"))

        # Reload and save to standardize intent codes and precision
        mri = MRIData.from_file(tmppath / f"{form}.nii.gz", dtype=np.double)
        logger.debug(f"Reloaded intermediate NIfTI file with shape {mri.data.shape} and dtype {mri.data.dtype}.")
        mri.save(outpath.with_suffix(".nii.gz"), dtype=np.single, intent_code=2001)
        logger.info(
            f"Final Look-Locker NIfTI saved to {outpath.with_suffix('.nii.gz')} with intent_code=2001 and dtype=np.single."
        )


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    subparser = parser.add_subparsers(dest="looklocker-command", help="Commands for processing Look-Locker data")

    dicom_parser = subparser.add_parser(
        "dcm2ll", help="Convert Look-Locker DICOM to NIfTI format", formatter_class=parser.formatter_class
    )
    dicom_parser.add_argument("-i", "--input", type=Path, help="Path to the input Look-Locker DICOM file")
    dicom_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the converted .nii.gz file")

    ll_timestamps = subparser.add_parser(
        "timestamps", help="Read timestamps from DICOM data", formatter_class=parser.formatter_class
    )
    ll_timestamps.add_argument("-i", "--input", type=Path, help="Path to the input Look-Locker DICOM file")
    ll_timestamps.add_argument("-o", "--output", type=Path, help="Desired output path for the generated file")

    ll_t1 = subparser.add_parser("t1", help="Generate a T1 map from Look-Locker data", formatter_class=parser.formatter_class)
    ll_t1.add_argument("-i", "--input", type=Path, help="Path to the 4D Look-Locker NIfTI file")
    ll_t1.add_argument("-t", "--timestamps", type=Path, help="Path to the text file containing trigger delay times (in ms)")
    ll_t1.add_argument("-o", "--output", type=Path, default=None, help="Path to save the resulting T1 map NIfTI file")

    ll_post = subparser.add_parser(
        "postprocess", help="Post-process a raw Look-Locker T1 map", formatter_class=parser.formatter_class
    )
    ll_post.add_argument("-i", "--input", type=Path, help="Path to the raw Look-Locker T1 map NIfTI file")
    ll_post.add_argument("-o", "--output", type=Path, default=None, help="Path to save the cleaned T1 map NIfTI file")
    ll_post.add_argument("--t1-low", type=float, default=100.0, help="Lower physiological limit for T1 values (in ms)")
    ll_post.add_argument("--t1-high", type=float, default=10000.0, help="Upper physiological limit for T1 values (in ms)")
    ll_post.add_argument(
        "--radius", type=int, default=10, help="Base radius for morphological dilation when generating the automatic mask"
    )
    ll_post.add_argument(
        "--erode-dilate-factor",
        type=float,
        default=1.3,
        help="Multiplier for the erosion radius relative to the dilation radius to ensure tight mask edges",
    )

    if extra_args_cb is not None:
        extra_args_cb(dicom_parser)
        extra_args_cb(ll_t1)
        extra_args_cb(ll_post)
        extra_args_cb(ll_timestamps)


def dispatch(args):
    command = args.pop("looklocker-command")
    if command == "dcm2ll":
        dicom_to_looklocker(args.pop("input"), args.pop("output"))
    elif command == "timestamps":
        timestamps = read_dicom_trigger_times(args.pop("input"))
        if args.pop("output") is not None:
            np.savetxt(args.pop("output"), timestamps)
    elif command == "t1":
        ll = LookLocker.from_file(args.pop("input"), args.pop("timestamps"))

        t1_map = ll.t1_map()

        output = args.pop("output")
        if output is not None:
            t1_map.mri.save(output, dtype=np.single)
            logger.info(f"Look-Locker T1 map saved to {output}.")
        else:
            logger.info("No output path provided, returning Look-Locker T1 map as MRIData object.")

    elif command == "postprocess":
        t1_map_post = LookLockerT1.from_file(args.pop("input")).postprocess(
            T1_low=args.pop("t1_low"),
            T1_high=args.pop("t1_high"),
            radius=args.pop("radius"),
            erode_dilate_factor=args.pop("erode_dilate_factor"),
        )
        output = args.pop("output")
        if output is not None:
            t1_map_post.save(output, dtype=np.single)
            logger.info(f"Post-processed Look-Locker T1 map saved to {output}.")
        else:
            logger.info("No output path provided, returning Post-processed Look-Locker T1 map as MRIData object.")
    else:
        raise ValueError(f"Unknown Look-Locker command: {command}")
