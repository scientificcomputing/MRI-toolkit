"""MRI Data orientation Module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
from .base import MRIData


def apply_affine(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply homogeneous-coordinate affine matrix T to each row of of matrix
    X of shape (N, 3)"""
    A = T[:-1, :-1]
    b = T[:-1, -1]
    return A.dot(X.T).T + b


def data_reorientation(mri_data: MRIData) -> MRIData:
    """Reorients the data-array and affine map such that the affine map is
    closest to the identity-matrix, such that increasing the first index
    corresponds to increasing the first coordinate in real space, and so on.

    The target coordinate system is still the same (i.e. RAS stays RAS)
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
    """Creates an affine map for change of coordinate system based on the
    string identifiers
     L(eft) <-> R(ight)
     P(osterior) <-> A(nterior)
     I(nferior) <-> S(uperior)
    change of coordinate system affine map"""
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
