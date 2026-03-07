from pathlib import Path
import numpy as np
import pytest

from mritk.statistics.compute_stats import extract_metadata, compute_region_statistics, generate_stats_dataframe
import mritk.cli as cli


def test_compute_stats_default(mri_data_dir: Path):
    seg_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    mri_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"

    dataframe = generate_stats_dataframe(seg_path, mri_path)

    assert not dataframe.empty
    assert set(dataframe.columns) == {
        "segmentation",
        "mri_data",
        "subject",
        "session",
        "timestamp",
        "label",
        "description",
        "voxelcount",
        "volume_ml",
        "num_nan_values",
        "sum",
        "mean",
        "median",
        "std",
        "min",
        "PC1",
        "PC5",
        "PC25",
        "PC75",
        "PC90",
        "PC95",
        "PC99",
        "max",
    }


def test_compute_stats_patterns(mri_data_dir: Path):
    seg_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    mri_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"

    dataframe = generate_stats_dataframe(
        seg_path,
        mri_path,
        seg_pattern=seg_pattern,
        mri_data_pattern=mri_data_pattern,
    )

    assert not dataframe.empty
    assert dataframe["subject"].iloc[0] == "sub-01"
    assert dataframe["segmentation"].iloc[0] == "aparc+aseg_refined"
    assert dataframe["mri_data"].iloc[0] == "concentration"
    assert dataframe["session"].iloc[0] == "ses-01"


def test_compute_stats_timestamp(mri_data_dir: Path):
    seg_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    mri_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    timetable = mri_data_dir / "timetable/timetable.tsv"
    timetable_sequence = "mixed"

    dataframe = generate_stats_dataframe(
        seg_path,
        mri_path,
        seg_pattern=seg_pattern,
        mri_data_pattern=mri_data_pattern,
        timestamp_path=timetable,
        timestamp_sequence=timetable_sequence,
    )

    assert dataframe["timestamp"].iloc[0] == -6414.9


def test_compute_stats_info(mri_data_dir: Path):
    seg_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    mri_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    info = {
        "mri_data": "concentration",
        "subject": "sub-01",
        "session": "ses-01",
        "segmentation": "aparc+aseg_refined",
    }

    dataframe = generate_stats_dataframe(seg_path, mri_path, info_dict=info)

    assert not dataframe.empty
    assert dataframe["subject"].iloc[0] == "sub-01"
    assert dataframe["segmentation"].iloc[0] == "aparc+aseg_refined"
    assert dataframe["mri_data"].iloc[0] == "concentration"
    assert dataframe["session"].iloc[0] == "ses-01"


def test_compute_mri_stats_cli(capsys, tmp_path: Path, mri_data_dir: Path):
    seg_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    mri_path = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    timetable = mri_data_dir / "timetable/timetable.tsv"
    timetable_sequence = "mixed"

    args = [
        "--segmentation",
        str(seg_path),
        "--mri",
        str(mri_path),
        "--output",
        str(tmp_path / "mri_stats_output.csv"),
        "--timetable",
        str(timetable),
        "--timelabel",
        timetable_sequence,
        "--seg_regex",
        seg_pattern,
        "--mri_regex",
        mri_data_pattern,
    ]

    ret = cli.main(["stats", "compute"] + args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Processing MRIs..." in captured.out
    assert "Stats successfully saved to" in captured.out
    assert (tmp_path / "mri_stats_output.csv").exists()


def test_extract_metadata_with_pattern():
    """Test extracting metadata successfully via regex pattern."""
    file_path = Path("sub-01_ses-01_concentration.nii.gz")
    pattern = r"(?P<subject>sub-\d{2})_(?P<session>ses-\d{2})_(?P<mri_data>[^\.]+)"

    info = extract_metadata(file_path, pattern=pattern)

    assert info["subject"] == "sub-01"
    assert info["session"] == "ses-01"
    assert info["mri_data"] == "concentration"


def test_extract_metadata_pattern_failure():
    """Test that a non-matching pattern correctly raises a RuntimeError."""
    file_path = Path("invalid_filename.nii.gz")
    pattern = r"(?P<subject>sub-\d{2})"

    with pytest.raises(RuntimeError, match="does not match the provided pattern"):
        extract_metadata(file_path, pattern=pattern)


def test_extract_metadata_with_info_dict():
    """Test fallback to info_dict when pattern is not provided."""
    file_path = Path("some_file.nii.gz")
    info_dict = {"subject": "sub-02", "segmentation": "aparc"}
    required_keys = ["subject", "segmentation", "mri_data"]

    info = extract_metadata(file_path, info_dict=info_dict, required_keys=required_keys)

    assert info["subject"] == "sub-02"
    assert info["segmentation"] == "aparc"
    assert info["mri_data"] is None  # Was not in info_dict


def test_compute_region_statistics_normal():
    """Test normal calculation of statistical metrics."""
    # Mock data: values 1.0 through 5.0
    region_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    labels = [10, 11]

    stats = compute_region_statistics(
        region_data=region_data, labels=labels, description="test_region", volscale=0.5, voxelcount=5
    )

    assert stats["description"] == "test_region"
    assert stats["label"] == "10,11"
    assert stats["voxelcount"] == 5
    assert stats["volume_ml"] == 2.5
    assert stats["num_nan_values"] == 0
    assert stats["sum"] == 15.0
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["median"] == 3.0


def test_compute_region_statistics_with_nans():
    """Test that statistics correctly ignore NaNs inside the region."""
    region_data = np.array([1.0, 2.0, np.nan, 3.0, np.nan])

    stats = compute_region_statistics(region_data=region_data, labels=[1], description="partial_nan", volscale=1.0, voxelcount=5)

    assert stats["num_nan_values"] == 2
    assert stats["sum"] == 6.0  # 1+2+3
    assert stats["mean"] == 2.0  # 6/3


def test_compute_region_statistics_empty_or_all_nan():
    """Test edge cases where the region is empty or completely composed of NaNs."""
    # Case 1: Empty (0 voxels)
    stats_empty = compute_region_statistics(
        region_data=np.array([]), labels=[1], description="empty_region", volscale=1.0, voxelcount=0
    )
    assert "mean" not in stats_empty
    assert stats_empty["voxelcount"] == 0

    # Case 2: All NaNs
    stats_nan = compute_region_statistics(
        region_data=np.array([np.nan, np.nan]), labels=[1], description="nan_region", volscale=1.0, voxelcount=2
    )
    assert stats_nan["num_nan_values"] == 2
    assert "mean" not in stats_nan
