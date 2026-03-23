from pathlib import Path

from mritk.statistics.metadata import (
    extract_metadata_from_bids,
    extract_pattern_from_path,
)


def test_path_extraction():
    pattern = r"sub-(?P<subject>[^\.]+)_(?P<session>ses-\d{2})_(?P<mri_data>[^\.]+)"

    path = Path("sub-01_ses-01_concentration.nii.gz")
    info = extract_pattern_from_path(pattern, path)
    assert info["subject"] == "01"
    assert info["session"] == "ses-01"
    assert info["mri_data"] == "concentration"


def test_bids_extraction():
    seg_path = Path("sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = Path("sub-01_ses-01_concentration.nii.gz")

    metadata = extract_metadata_from_bids(segmentation_path=seg_path, mri_data_path=mri_path)

    assert metadata["subject"] == "01"
    assert metadata["session"] == "ses-01"
    assert metadata["mri_data"] == "concentration"
    assert metadata["segmentation"] == "aparc+aseg_refined"
