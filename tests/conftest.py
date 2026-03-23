import os
from pathlib import Path

import numpy as np
import pytest

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
