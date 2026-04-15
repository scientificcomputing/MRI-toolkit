import os
import typing
from pathlib import Path

import numpy as np
import pytest

import mritk
from mritk.data import MRIData
from mritk.segmentation import Segmentation


@pytest.fixture(scope="session")
def mri_data_dir() -> Path:
    return Path(os.getenv("MRITK_TEST_DATA_FOLDER", "test_data"))


@pytest.fixture
def example_segmentation() -> Segmentation:
    """Example segmentation"""
    base = np.array([0, 1, 2, 3], dtype=float)
    seg = np.tile(base, (100, 1))

    return Segmentation(seg, affine=np.eye(4))


@pytest.fixture
def example_values() -> MRIData:
    """Example values for testing qoi computations"""
    np.random.seed(0)

    data = np.array(
        [
            np.random.normal(0.0, size=100),
            np.random.normal(1.0, size=100),
            np.random.normal(2.0, size=100),
            np.random.normal(3.0, size=100),
        ]
    ).T
    return MRIData(data, affine=np.eye(4))


class GonzoRoi(typing.NamedTuple):
    points: np.ndarray
    affine: np.ndarray
    shape: tuple

    def voxel_indices(self, affine: np.ndarray) -> np.ndarray:
        return mritk.data.physical_to_voxel_indices(self.points, affine=affine)


@pytest.fixture
def gonzo_roi() -> GonzoRoi:
    xs = np.linspace(0, 70, 80)
    ys = np.linspace(0, 20, 20)
    zs = np.linspace(-40, 90, 110)

    # Create a 3D grid of points as one long vector
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T  # Shape: (N, 3)
    grid_shape = grid_x.shape

    new_affine = np.eye(4)
    new_affine[0, 0] = xs[1] - xs[0]
    new_affine[1, 1] = ys[1] - ys[0]
    new_affine[2, 2] = zs[1] - zs[0]
    new_affine[0, 3] = xs[0]
    new_affine[1, 3] = ys[0]
    new_affine[2, 3] = zs[0]
    return GonzoRoi(points=grid_points, affine=new_affine, shape=grid_shape)
