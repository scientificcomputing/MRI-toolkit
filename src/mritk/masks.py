# Intracranial and CSF masks generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.interpolate
import skimage

from .data import MRIData
from .testing import assert_same_space


def largest_island(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Identifies and returns the largest contiguous region (island) in a boolean mask.

    Args:
        mask (np.ndarray): A boolean or integer array where non-zero values
            represent the regions of interest.
        connectivity (int, optional): Maximum number of orthogonal hops to consider
            a pixel/voxel as connected. For 2D, 1=4-connected, 2=8-connected.
            For 3D, 1=6-connected, 2=18-connected, 3=26-connected. Defaults to 1.

    Returns:
        np.ndarray: A boolean array of the same shape as `mask`, where True
        indicates the elements of the largest connected component.
    """
    newmask = skimage.measure.label(mask, connectivity=connectivity)
    regions = skimage.measure.regionprops(newmask)

    # Handle the edge case where the mask is completely empty
    if not regions:
        return np.zeros_like(mask, dtype=bool)

    regions.sort(key=lambda x: x.num_pixels, reverse=True)
    return newmask == regions[0].label


def compute_csf_mask_array(
    vol: np.ndarray,
    connectivity: int | None = 2,
    use_li: bool = False,
) -> np.ndarray:
    """
    Creates a binary mask isolating the Cerebrospinal Fluid (CSF).

    This function uses intensity thresholding (either Li or Yen) to separate
    bright fluid regions from surrounding tissue. It then isolates the CSF
    by retaining only the largest contiguous spatial island.

    Args:
        vol (np.ndarray): 3D numpy array of the MRI volume (typically T2-weighted or Spin-Echo).
        connectivity (Optional[int], optional): Maximum connectivity distance to evaluate
            contiguous islands. Defaults to 2.
        use_li (bool, optional): If True, uses Li's minimum cross entropy thresholding.
            If False, uses Yen's thresholding based on the volume histogram. Defaults to False.

    Returns:
        np.ndarray: A boolean 3D array representing the CSF mask.
    """
    connectivity = connectivity or vol.ndim

    if use_li:
        thresh = skimage.filters.threshold_li(vol)
        binary = vol > thresh
        binary = largest_island(binary, connectivity=connectivity)
    else:
        # Create a histogram excluding the absolute background (0) and extreme high outliers
        valid_mask = (vol > 0) & (vol < np.quantile(vol, 0.999))
        hist, bins = np.histogram(vol[valid_mask], bins=512)

        thresh = skimage.filters.threshold_yen(hist=(hist, bins))
        binary = vol > thresh
        binary = largest_island(binary, connectivity=connectivity)

    return binary


def csf_mask(input: Path, connectivity: int | None = 2, use_li: bool = False) -> MRIData:
    """
    I/O wrapper for generating and saving a CSF mask from a NIfTI file.

    Args:
        input (Path): Path to the input NIfTI image.
        connectivity (Optional[int], optional): Connectivity distance. Defaults to 2.
        use_li (bool, optional): If True, uses Li thresholding. Defaults to False.
        output (Optional[Path], optional): Path to save the resulting mask. Defaults to None.

    Returns:
        MRIData: An MRIData object containing the boolean mask array.

    Raises:
        AssertionError: If the resulting mask contains no voxels.
    """
    input_vol = MRIData.from_file(input, dtype=np.single)
    mask = compute_csf_mask_array(input_vol.data, connectivity, use_li)

    assert np.max(mask) > 0, "Masking failed, no voxels in mask"

    mri_data = MRIData(data=mask, affine=input_vol.affine)

    return mri_data


def csf_segmentation(input_segmentation: Path | MRIData, csf_mask: Path | MRIData) -> MRIData:
    """
    Generates a CSF segmentation by applying a CSF mask to an anatomical segmentation.

    This function takes an anatomical segmentation (e.g., from FreeSurfer) and a CSF mask,
    and produces a new segmentation where voxels identified as CSF in the mask are labeled
    with their original segmentation values, while non-CSF voxels are set to zero.

    Args:
        input_segmentation (Path | MRIData): Path to the anatomical segmentation NIfTI file
            or an MRIData object containing the resampled segmentation.
        csf_mask (Path | MRIData): Either a path to a CSF mask NIfTI file or an MRIData object containing the mask.

    Returns:
        MRIData: An MRIData object containing the CSF segmentation.
    """
    if isinstance(input_segmentation, Path):
        seg_mri = MRIData.from_file(input_segmentation, dtype=np.int16)
    else:
        seg_mri = input_segmentation

    if isinstance(csf_mask, Path):
        csf_mask_mri = MRIData.from_file(csf_mask, dtype=bool)
    else:
        csf_mask_mri = csf_mask

    assert_same_space(seg_mri, csf_mask_mri)

    # Get interpolation operator
    I, J, K = np.where(seg_mri.data != 0)
    interp = scipy.interpolate.NearestNDInterpolator(np.array([I, J, K]).T, seg_mri.data[I, J, K])
    # Interpolate segmentation values at CSF mask locations
    i, j, k = np.where(csf_mask_mri.data != 0)
    csf_seg = np.zeros_like(seg_mri.data, dtype=np.int16)
    csf_seg[i, j, k] = interp(i, j, k)

    return MRIData(data=csf_seg.astype(np.int16), affine=csf_mask_mri.affine)


def compute_intracranial_mask_array(csf_mask_array: np.ndarray, segmentation_array: np.ndarray) -> np.ndarray:
    """
    Combines a CSF mask array and a brain segmentation mask array into a solid intracranial mask.

    This function merges the two domains and uses morphological operations (binary opening)
    on the background to cleanly fill in any gaps or holes within the intracranial space.

    Args:
        csf_mask_array (np.ndarray): 3D boolean array representing the CSF mask.
        segmentation_array (np.ndarray): 3D boolean array representing the anatomical brain segmentation.

    Returns:
        np.ndarray: A boolean 3D array representing the solid intracranial space.
    """
    # Ensure logical boolean combination
    combined_mask = csf_mask_array.astype(bool) | segmentation_array.astype(bool)

    # Identify the background by extracting the largest island of the inverted combined mask
    background_mask = largest_island(~combined_mask, connectivity=1)

    # Smooth the background boundary to fill narrow sulci/gaps
    opened_background = skimage.morphology.opening(background_mask, skimage.morphology.ball(3))

    # The intracranial mask is the inverse of the cleaned background
    return ~opened_background


def intracranial_mask(segmentation_path: Path, csf_mask_path: Path) -> MRIData:
    """
    I/O wrapper for generating and saving an intracranial mask from NIfTI files.

    Loads the masks, verifies they share the same physical coordinate space, and
    delegates the array computation.

    Args:
        segmentation_path (Path): Path to the brain (refined) segmentation NIfTI file, \
            generated by the segmentation refinement module.
        csf_mask_path (Path): Path to the CSF mask, generated by the csf mask module.

    Returns:
        MRIData: An MRIData object containing the intracranial mask.
    """
    # Get segmentation data and csf segmentation
    segmentation_data = MRIData.from_file(segmentation_path, dtype=bool)
    csf_seg = csf_segmentation(input_segmentation=segmentation_data, csf_mask=csf_mask_path)

    # Validate spatial alignment before array operations
    assert_same_space(csf_seg, segmentation_data)

    mask_data = compute_intracranial_mask_array(csf_seg.data, segmentation_data.data)
    mri_data = MRIData(data=mask_data, affine=segmentation_data.affine)

    return mri_data


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    subparser = parser.add_subparsers(dest="mask-command", help="Commands for generating mask")

    csf_mask_parser = subparser.add_parser("csf", help="Compute CSF mask", formatter_class=parser.formatter_class)
    csf_mask_parser.add_argument("-i", "--input", type=Path, help="Path to the input NIfTI image - usually a T2-weighted")
    csf_mask_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the resulting mask")
    csf_mask_parser.add_argument(
        "--connectivity", type=int, default=2, help="Maximum connectivity distance to evaluate contiguous islands"
    )
    csf_mask_parser.add_argument(
        "--use-li",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="--use-li: Li thresholding, --no-use-li: Yen thresholding",
    )

    intracranial_mask_parser = subparser.add_parser(
        "intracranial", help="Compute intracranial mask", formatter_class=parser.formatter_class
    )
    intracranial_mask_parser.add_argument(
        "--segmentation-path",
        type=Path,
        help="Path to refined segmentation file, generated by \
        the segmentation refinement module, i.e. mritk seg refine",
    )
    intracranial_mask_parser.add_argument(
        "--csf-mask-path",
        type=Path,
        help="Path to the CSF mask NIfTI file, generated by \
        the csf mask module, i.e. mritk mask csf",
    )
    intracranial_mask_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the resulting mask")

    if extra_args_cb is not None:
        extra_args_cb(csf_mask_parser)
        extra_args_cb(intracranial_mask_parser)


def dispatch(args):
    command = args.pop("mask-command")
    if command == "csf":
        csf_mask_data = csf_mask(input=args.pop("input"), connectivity=args.pop("connectivity"), use_li=args.pop("use_li"))
        csf_mask_data.save(args.pop("output"), dtype=np.uint8)
    elif command == "intracranial":
        intracranial_mask_data = intracranial_mask(
            segmentation_path=args.pop("segmentation_path"), csf_mask_path=args.pop("csf_mask_path")
        )
        intracranial_mask_data.save(args.pop("output"), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown mask command: {command}")
