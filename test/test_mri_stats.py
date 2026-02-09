"""MRI Stats - Test

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path


from mritk.statistics.compute_stats import generate_stats_dataframe  # , compute_mri_stats
import mritk.cli as cli


def test_compute_stats_default(mri_data_dir: Path):
    seg_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )
    mri_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "concentrations/sub-01_ses-01_concentration.nii.gz"
    )

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
    seg_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )
    mri_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "concentrations/sub-01_ses-01_concentration.nii.gz"
    )
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = (
        "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    )

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
    seg_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )
    mri_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "concentrations/sub-01_ses-01_concentration.nii.gz"
    )
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = (
        "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    )
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
    seg_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )
    mri_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "concentrations/sub-01_ses-01_concentration.nii.gz"
    )
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
    seg_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )
    mri_path = (
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01"
        / "concentrations/sub-01_ses-01_concentration.nii.gz"
    )
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = (
        "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    )
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
