"""MRI Stats - Test

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import os
from click.testing import CliRunner
from pathlib import Path

from MRI.statistics.compute_stats import generate_stats_dataframe, compute_mri_stats
import pytest
import download_data

links = {
    "mri-processed.zip": "https://zenodo.org/records/14266867/files/mri-processed.zip?download=1",
    "timetable.tsv": "https://github.com/jorgenriseth/gonzo/blob/main/mri_dataset/timetable.tsv?raw=true"
}

@pytest.fixture(scope="session")
def mri_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mri_test_data")
    download_data.download_multiple(links, data_dir)
    
    return data_dir

def test_compute_stats_default(mri_data_dir):
    seg_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz")

    dataframe = generate_stats_dataframe(
        seg_path,
        mri_path
    )

    assert not dataframe.empty
    assert set(dataframe.columns) == {'segmentation', 'mri_data', 'subject', 'session', 'timestamp', 'label',
                                      'description', 'voxelcount', 'volume_ml', 'num_nan_values', 'sum',
                                      'mean', 'median', 'std', 'min', 'PC1', 'PC5', 'PC25', 'PC75', 'PC90',
                                      'PC95', 'PC99', 'max'}


def test_compute_stats_patterns(mri_data_dir):
    seg_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz")
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


def test_compute_stats_timestamp(mri_data_dir):
    seg_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz")
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    timetable = os.path.join(mri_data_dir, "timetable/timetable.tsv")
    timetable_sequence = "mixed"

    dataframe = generate_stats_dataframe(
        seg_path,
        mri_path,
        seg_pattern=seg_pattern,
        mri_data_pattern=mri_data_pattern,
        timestamp_path=timetable,
        timestamp_sequence=timetable_sequence
    )

    assert dataframe["timestamp"].iloc[0] == -6414.9


def test_compute_stats_info(mri_data_dir):
    seg_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz")
    info = {"mri_data": "concentration", "subject": "sub-01", "session": "ses-01", "segmentation": "aparc+aseg_refined"}

    dataframe = generate_stats_dataframe(
        seg_path,
        mri_path,
        info_dict=info
    )

    assert not dataframe.empty
    assert dataframe["subject"].iloc[0] == "sub-01"
    assert dataframe["segmentation"].iloc[0] == "aparc+aseg_refined"
    assert dataframe["mri_data"].iloc[0] == "concentration"
    assert dataframe["session"].iloc[0] == "ses-01"


def test_compute_mri_stats_cli(tmp_path, mri_data_dir):
    runner = CliRunner()
    seg_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz")
    mri_path = os.path.join(mri_data_dir, "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz")
    seg_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_seg-(?P<segmentation>[^\\.]+)"
    mri_data_pattern = "(?P<subject>sub-(control|patient)*\\d{2})_(?P<session>ses-\\d{2})_(?P<mri_data>[^\\.]+)"
    timetable = os.path.join(mri_data_dir, "timetable/timetable.tsv")
    timetable_sequence = "mixed"

    result = runner.invoke(
        compute_mri_stats,
        [
            "--segmentation", seg_path,
            "--mri", mri_path,
            "--output", Path(str(tmp_path / "mri_stats_output.csv")),
            "--timetable", timetable,
            "--timelabel", timetable_sequence,
            "--seg_regex", seg_pattern,
            "--mri_regex", mri_data_pattern
        ]
    )
    assert result.exit_code == 0