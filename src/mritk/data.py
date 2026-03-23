# MRI Data Base class and functions Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


from pathlib import Path
from typing import Optional

import nibabel
import numpy as np
import numpy.typing as npt


def load_mri_data(path: Path | str, dtype: npt.DTypeLike | None = None, orient: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load MRI data from a file and return the data array and affine matrix.

    Args:
        path: Path to the MRI file.
        dtype: Data type for the returned array.
        orient: Whether to reorient the data.

    Returns:
        Tuple of (data, affine) arrays.
    """
    filepath = Path(path)
    suffix = check_suffix(filepath)
    if suffix in (".nii", ".nii.gz"):
        mri = nibabel.nifti1.load(filepath)
    elif suffix in (".mgz", ".mgh"):
        mri = nibabel.freesurfer.mghformat.load(filepath)
    else:
        raise ValueError(f"Invalid suffix {filepath}, should be either '.nii', or '.mgz'")

    affine = mri.affine
    if affine is None:
        raise RuntimeError("MRI do not contain affine")

    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    data = np.asarray(mri.get_fdata("unchanged"), **kwargs)

    if orient:
        data, affine = data_reorientation(data, affine)

    return data, affine


def save_mri_data(data: np.ndarray, affine: np.ndarray, path: Path | str, intent_code: Optional[int] = None):
    """Save MRI data to a file.

    Args:
        data: The MRI data array to save.
        affine: The affine transformation matrix associated with the data.
        path: Path to the file to save.
        dtype: Data type for the saved array.
        intent_code: Intent code for the saved file.
    """
    save_path = Path(path)
    suffix = check_suffix(save_path)
    if suffix in (".nii", ".nii.gz"):
        nii = nibabel.nifti1.Nifti1Image(data, affine)
        if intent_code is not None:
            nii.header.set_intent(intent_code)
        nibabel.nifti1.save(nii, save_path)
    elif suffix in (".mgz", ".mgh"):
        mgh = nibabel.freesurfer.mghformat.MGHImage(data, affine)
        if intent_code is not None:
            mgh.header.set_intent(intent_code)
        nibabel.freesurfer.mghformat.save(mgh, save_path)
    else:
        raise ValueError(f"Invalid suffix {save_path}, should be either '.nii', or '.mgz'")


def check_suffix(filepath: Path):
    suffix = filepath.suffix
    if suffix == ".gz":
        suffixes = filepath.suffixes
        if len(suffixes) >= 2 and suffixes[-2] == ".nii":
            return ".nii.gz"
    return suffix


class MRIData:
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        self.data = data
        self.affine = affine

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.affine

    @property
    def voxel_ml_volume(self) -> float:
        # Calculate the volume of a single voxel in milliliters
        voxel_volume_mm3 = abs(np.linalg.det(self.affine[:3, :3]))
        voxel_volume_ml = voxel_volume_mm3 / 1000.0  # Convert from mm^3 to ml
        return voxel_volume_ml

    @classmethod
    def from_file(cls, path: Path | str, dtype: npt.DTypeLike | None = None, orient: bool = True) -> "MRIData":
        data, affine = load_mri_data(path, dtype=dtype, orient=orient)
        return cls(data=data, affine=affine)

    def save(self, path: Path | str, dtype: npt.DTypeLike | None = None, intent_code: Optional[int] = None):
        if dtype is None:
            dtype = self.data.dtype
        data = self.data.astype(dtype)
        save_mri_data(data, self.affine, path, intent_code=intent_code)


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


def data_reorientation(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient the data array and affine matrix to the canonical orientation.

    This function adjusts the data array layout (via transpositions and flips)
    so that the affine matrix becomes as close to the identity matrix as possible.
    This ensures that increasing array indices correspond to increasing spatial
    coordinates in the physical space.

    Note:
        The physical coordinate system remains unchanged (e.g., RAS stays RAS).

    Args:
        data: The input MRI data array.
        affine: The input affine transformation matrix.

    Returns:
        A tuple containing the reoriented data array and updated affine matrix.
    """
    A = affine[:3, :3]
    flips = np.sign(A[np.argmax(np.abs(A), axis=0), np.arange(3)]).astype(int)
    permutes = np.argmax(np.abs(A), axis=0)
    offsets = ((1 - flips) // 2) * (np.array(data.shape[:3]) - 1)

    # Index flip matrix
    F = np.eye(4, dtype=int)
    F[:3, :3] = np.diag(flips)
    F[:3, 3] = offsets

    # Index permutation matrix
    P = np.eye(4, dtype=int)[[*permutes, 3]]
    affine = affine @ F @ P
    inverse_permutes = np.argmax(P[:3, :3].T, axis=1)
    data = data[:: flips[0], :: flips[1], :: flips[2], ...].transpose([*inverse_permutes, *list(range(3, data.ndim))]).copy()
    return data, affine


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
