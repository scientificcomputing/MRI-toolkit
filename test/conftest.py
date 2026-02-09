import os
import pytest


@pytest.fixture(scope="session")
def mri_data_dir():
    return os.getenv("MRITK_TEST_DATA_FOLDER", "test_data")
