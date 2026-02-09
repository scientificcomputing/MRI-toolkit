from pathlib import Path
import os
import pytest


@pytest.fixture(scope="session")
def mri_data_dir() -> Path:
    return Path(os.getenv("MRITK_TEST_DATA_FOLDER", "test_data"))
