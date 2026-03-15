from pathlib import Path

import numpy as np

from .data import MRIData


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

    img1 = MRIData.from_file(img_path1, orient=False)
    img2 = MRIData.from_file(img_path2, orient=False)

    # 1. Compare Image Data
    data1 = img1.data
    data2 = img2.data

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


def assert_same_space(mri1: MRIData, mri2: MRIData, rtol: float = 1e-5):
    """Assert that two MRI datasets share the same physical space.

    Checks if the data shapes are identical and if the affine transformation
    matrices are close within a specified relative tolerance.

    Args:
        mri1: The first MRI data object.
        mri2: The second MRI data object.
        rtol: Relative tolerance for comparing affine matrices. Defaults to 1e-5.

    Raises:
        ValueError: If shapes differ or if affine matrices are not sufficiently close.
    """
    if mri1.data.shape == mri2.data.shape and np.allclose(mri1.affine, mri2.affine, rtol):
        return
    with np.printoptions(precision=5):
        err = np.nanmax(np.abs((mri1.affine - mri2.affine) / mri2.affine))
        msg = (
            f"MRI's not in same space (relative tolerance {rtol})."
            f" Shapes: ({mri1.data.shape}, {mri2.data.shape}),"
            f" Affines: {mri1.affine}, {mri2.affine},"
            f" Affine max relative error: {err}"
        )

        raise ValueError(msg)
