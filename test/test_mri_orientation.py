"""MRI Orientation - Test

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
from mri.data.orientation import apply_affine, change_of_coordinates_map


def default_test_data():
    return np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 4, 3],
                [0, 2, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=float,
    )


def default_test_affine():
    return np.array(
        [
            [10, 0, 0, -10],
            [0, 10, 0, -9],
            [0, 0, 10, -11],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def test_change_of_coordinates():
    A, T = default_test_data(), default_test_affine()
    IJK = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 1, 0],
        ]
    )
    xyz = np.array(
        [
            [-10, -9, -11],
            [0, 1, -1],
            [10, 11, 9],
            [10, 1, -11],
        ]
    )
    assert np.allclose(apply_affine(T, IJK), xyz)

    ras2lia = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    assert np.allclose(change_of_coordinates_map("RAS", "LIA"), ras2lia)

    lia_xyz = np.array(
        [
            [10, 11, -9],
            [0, 1, 1],
            [-10, -9, 11],
            [-10, 11, 1],
        ]
    )
    assert np.allclose(apply_affine(ras2lia @ T, IJK), lia_xyz)
