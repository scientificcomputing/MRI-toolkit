from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mritk.segmentation import (
    LUT_REGEX,
    VENTRICLES,
    ExtendedFreeSurferSegmentation,
    Segmentation,
    default_segmentation_groups,
    lut_record,
    read_lut,
    resolve_lut_path,
    validate_lut_file,
    write_lut,
)


def test_segmentation_initialization(example_segmentation: Segmentation):
    assert example_segmentation.data.shape == (100, 4)
    assert example_segmentation.affine.shape == (4, 4)
    assert example_segmentation.num_rois == 3
    assert set(example_segmentation.roi_labels) == {1, 2, 3}
    assert example_segmentation.lut.shape == (3, 1)
    assert set(example_segmentation.lut.columns) == {"Label"}


def test_freesurfer_segmentation_labels(mri_data_dir: Path):
    fs_seg = ExtendedFreeSurferSegmentation.from_file(
        mri_data_dir
        / "mri-processed"
        / "mri_processed_data"
        / "sub-01"
        / "segmentations"
        / "sub-01_seg-aparc+aseg_refined.nii.gz"
    )

    labels = fs_seg.get_roi_labels()
    assert not labels.empty
    assert set(labels["ROI"]) == set(fs_seg.roi_labels)


def test_extended_freesurfer_segmentation_labels(example_segmentation: Segmentation, mri_data_dir: Path):
    data = example_segmentation.data
    data[0:2, 0:2] = 10001  # csf
    data[3:5, 3:5] = 20001  # dura

    ext_fs_seg = ExtendedFreeSurferSegmentation(data, affine=np.eye(4))
    labels = ext_fs_seg.get_roi_labels()

    assert set(labels["ROI"]) == set(ext_fs_seg.roi_labels)
    assert labels.loc[labels["ROI"] == 10001, "tissue_type"].iloc[0] == "CSF"
    assert labels.loc[labels["ROI"] == 20001, "tissue_type"].iloc[0] == "Dura"
    assert labels.loc[labels["ROI"] == 10001, "Label"].iloc[0] == labels.loc[labels["ROI"] == 1, "Label"].iloc[0]
    assert labels.loc[labels["ROI"] == 20001, "Label"].iloc[0] == labels.loc[labels["ROI"] == 1, "Label"].iloc[0]


def test_default_segmentation_groups():
    """Test that the segmentation groups return the expected predefined structures."""
    groups = default_segmentation_groups()

    assert "cerebral-wm" in groups
    assert "csf-freesurfer" in groups
    assert isinstance(groups["cerebral-wm"], list)

    # Ventricles should be part of the freesurfer CSF list
    for v in VENTRICLES:
        assert v in groups["csf-freesurfer"]


def test_lut_record_parsing():
    """Test that the regex parser properly extracts and normalizes RGBA values."""
    # FreeSurfer format: Label, Name, R, G, B, A
    line = "4   Left-Lateral-Ventricle    120   18   134   0"
    match = LUT_REGEX.match(line)

    assert match is not None
    record = lut_record(match)

    assert record["label"] == 4
    assert record["description"] == "Left-Lateral-Ventricle"
    assert record["R"] == 120 / 255.0
    assert record["G"] == 18 / 255.0
    assert record["B"] == 134 / 255.0
    # Alpha in FreeSurfer is inverted (0 = opaque). Our parsed record should be 1.0 (opaque).
    assert record["A"] == 1.0


def test_validate_lut_file_valid(tmp_path):
    """Test that a properly formatted LUT file is correctly validated."""
    valid_file = tmp_path / "valid_lut.txt"
    valid_file.write_text("# This is a comment\n2   Left-Cerebral-White-Matter      245 245 245 0\n")
    assert validate_lut_file(valid_file) is True


def test_validate_lut_file_invalid(tmp_path):
    """Test that an improperly formatted file fails validation."""
    invalid_file = tmp_path / "invalid_lut.txt"
    invalid_file.write_text("Just some random text\nNo valid records here.")
    assert validate_lut_file(invalid_file) is False


def test_validate_lut_file_empty(tmp_path):
    """Test that an empty file fails validation."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert validate_lut_file(empty_file) is False


def test_resolve_lut_path_existing_invalid_raises_error(tmp_path):
    """Test that providing an existing but invalid file raises a ValueError."""
    invalid_file = tmp_path / "bad_lut.txt"
    invalid_file.write_text("Not a LUT file.")

    with pytest.raises(ValueError, match="invalid or corrupted"):
        resolve_lut_path(invalid_file)


def test_resolve_lut_path_custom_target_download(tmp_path):
    """Test that missing custom files trigger a download to the specified custom path."""
    custom_target = tmp_path / "my_custom_folder" / "my_lut.txt"

    # File does not exist yet. It should be downloaded directly to `custom_target`
    resolved_path = resolve_lut_path(custom_target)

    assert resolved_path == custom_target
    assert custom_target.exists()
    assert validate_lut_file(custom_target) is True


def test_resolve_lut_path_default_download(tmp_path):
    """Test that if no file is provided, it downloads to the default location."""
    with patch("os.environ", {}), patch("pathlib.Path.cwd", return_value=tmp_path):
        resolved_path = resolve_lut_path(None)

        expected_target = tmp_path / "FreeSurferColorLUT.txt"
        assert resolved_path == expected_target
        assert expected_target.exists()
        assert validate_lut_file(expected_target) is True


def test_read_lut_file_io(tmp_path):
    """Test reading a real LUT file written to disk."""
    dummy_lut_file = tmp_path / "dummy_lut.txt"
    dummy_lut_file.write_text(
        "# This is a comment\n"
        "2   Left-Cerebral-White-Matter      245 245 245 0\n"
        "3   Left-Cerebral-Cortex            205 62  78  0\n"
    )

    df = read_lut(dummy_lut_file)

    assert len(df) == 2
    assert df.iloc[0]["label"] == 2
    assert df.iloc[0]["description"] == "Left-Cerebral-White-Matter"
    assert df.iloc[1]["label"] == 3
    assert df.iloc[1]["R"] == 205 / 255.0


def test_write_lut_file_io(tmp_path):
    """Test saving a DataFrame back to the FreeSurfer format."""
    dummy_lut_file = tmp_path / "saved_lut.txt"

    # Mock DataFrame matching the parsed structure (normalized floats)
    data = [
        {"label": 4, "description": "Left-Lateral-Ventricle", "R": 120 / 255.0, "G": 18 / 255.0, "B": 134 / 255.0, "A": 1.0},
        {"label": 5, "description": "Left-Inf-Lat-Vent", "R": 198 / 255.0, "G": 51 / 255.0, "B": 122 / 255.0, "A": 1.0},
    ]
    df = pd.DataFrame(data)

    write_lut(dummy_lut_file, df)

    assert dummy_lut_file.exists()
    content = dummy_lut_file.read_text().splitlines()

    assert len(content) == 2
    # Verify the denormalization restored the original 0-255 integers
    assert content[0] == "4\tLeft-Lateral-Ventricle\t120\t18\t134\t0"
    assert content[1] == "5\tLeft-Inf-Lat-Vent\t198\t51\t122\t0"
