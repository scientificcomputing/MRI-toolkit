# MRI Data orientation Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np

from .base import MRIData


def physical_to_voxel_indices(physical_coordinates: np.ndarray, affine: np.ndarray, round_coords: bool = True) -> np.ndarray:
    """Transform physical coordinates to voxel indices using an affine matrix.

    This function maps coordinates from physical space (e.g., FEM degrees of freedom)
    back to the image voxel space by applying the inverse of the provided affine
    transformation matrix.

    Args:
        physical_coordinates: (N, 3) array of coordinates in physical space (world coordinates).
        affine: (4, 4) affine transformation matrix mapping voxel indices to physical space.
        round_coords: If True, rounds the resulting voxel coordinates to the nearest
            integer and casts them to `int`. If False, returns floating-point voxel coordinates.
            Defaults to True.

    Returns:
        (N, 3) array of voxel indices (or coordinates).
    """
    # Note: Assumes apply_affine is available in the scope or imported
    img_space_coords = apply_affine(np.linalg.inv(affine), physical_coordinates)
    if round_coords:
        return np.rint(img_space_coords).astype(int)
    return img_space_coords


def find_nearest_valid_voxels(query_indices: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Find the nearest valid voxels in a mask for a set of query indices.

    Uses a KDTree to find the `k` nearest neighbors for each point in `query_indices`
    where the neighbors are restricted to positions where `mask` is True.

    Args:
        query_indices: (N, 3) array of voxel indices (or coordinates) to find neighbors for.
        mask: Boolean array of shape (X, Y, Z). Neighbors will only be selected from
            coordinates where this mask is True.
        k: The number of nearest neighbors to find for each query point.

    Returns:
        Array of nearest neighbor indices.
        - If k=1: Returns shape (3, 1, N) containing the coordinates of the single nearest neighbor.
        - If k>1: Returns shape (3, k, N) containing the coordinates of the k nearest neighbors.

    Raises:
        ValueError: If the provided mask contains no valid (True) entries.
    """
    import scipy.spatial

    valid_inds = np.argwhere(mask)
    if len(valid_inds) == 0:
        raise ValueError("No valid indices found in mask.")

    tree = scipy.spatial.KDTree(valid_inds)
    _, indices = tree.query(query_indices, k=k)

    # Transpose to match the expected output shape (3, k, N) or (3, 1, N)
    dof_neighbours = valid_inds[indices].T

    if k == 1:
        dof_neighbours = dof_neighbours[:, np.newaxis, :]
    return dof_neighbours


def apply_affine(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a homogeneous affine transformation matrix to a set of points.

    Args:
        T: (4, 4) affine transformation matrix.
        X: (N, 3) matrix of points, where each row is a point (x, y, z).

    Returns:
        (N, 3) matrix of transformed points.
    """
    A = T[:-1, :-1]
    b = T[:-1, -1]
    return A.dot(X.T).T + b


def data_reorientation(mri_data: MRIData) -> MRIData:
    """Reorient the data array and affine matrix to the canonical orientation.

    This function adjusts the data array layout (via transpositions and flips)
    so that the affine matrix becomes as close to the identity matrix as possible.
    This ensures that increasing array indices correspond to increasing spatial
    coordinates in the physical space.

    Note:
        The physical coordinate system remains unchanged (e.g., RAS stays RAS).

    Args:
        mri_data: The input MRI data object containing the data array and affine.

    Returns:
        A new MRIData object with reoriented data and updated affine matrix.
    """
    A = mri_data.affine[:3, :3]
    flips = np.sign(A[np.argmax(np.abs(A), axis=0), np.arange(3)]).astype(int)
    permutes = np.argmax(np.abs(A), axis=0)
    offsets = ((1 - flips) // 2) * (np.array(mri_data.data.shape[:3]) - 1)

    # Index flip matrix
    F = np.eye(4, dtype=int)
    F[:3, :3] = np.diag(flips)
    F[:3, 3] = offsets

    # Index permutation matrix
    P = np.eye(4, dtype=int)[[*permutes, 3]]
    affine = mri_data.affine @ F @ P
    inverse_permutes = np.argmax(P[:3, :3].T, axis=1)
    data = (
        mri_data.data[:: flips[0], :: flips[1], :: flips[2], ...]
        .transpose([*inverse_permutes, *list(range(3, mri_data.data.ndim))])
        .copy()
    )
    return MRIData(data, affine)


def change_of_coordinates_map(orientation_in: str, orientation_out: str) -> np.ndarray:
    """Create an affine transformation map between two coordinate systems.

    This generates a 4x4 affine matrix that maps coordinates from the `orientation_in`
    system to the `orientation_out` system based on standard anatomical axis labels.

    Supported Labels:
        - L (Left) <-> R (Right)
        - P (Posterior) <-> A (Anterior)
        - I (Inferior) <-> S (Superior)

    Args:
        orientation_in: String defining the source orientation (e.g., "RAS").
        orientation_out: String defining the target orientation (e.g., "LIA").

    Returns:
        (4, 4) affine transformation matrix.

    Raises:
        ValueError: If an invalid axis label is provided or if axes cannot be matched.
    """
    axes_labels = {
        "R": 0,
        "L": 0,
        "A": 1,
        "P": 1,
        "S": 2,
        "I": 2,
    }
    order = np.nan * np.empty(len(orientation_in))
    for idx1, char1 in enumerate(orientation_in):
        if char1 not in axes_labels:
            raise ValueError(f"'{char1}' not a valid axis label")

        # Start indexing at 1 to avoid 0 in the sign-function.
        for idx2, char2 in enumerate(orientation_out, start=1):
            if char2 not in axes_labels:
                raise ValueError(f"'{char2}' not a valid axis label")
            if axes_labels[char1] == axes_labels[char2]:
                if char1 == char2:
                    order[idx1] = idx2
                else:
                    order[idx1] = -idx2
                break

            if idx2 == len(orientation_out):
                print(char1, char2)
                raise ValueError(f"Couldn't find axis in '{orientation_out}' corresponding to '{char1}'")
    index_flip = np.sign(order).astype(int)
    index_order = np.abs(order).astype(int) - 1  # Map back to 0-indexing

    F = np.eye(4)
    F[:3, :3] = np.diag(index_flip)

    P = np.eye(4)
    P[:, :3] = P[:, index_order]
    return P @ F


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
