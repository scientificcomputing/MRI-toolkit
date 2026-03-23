import re
from pathlib import Path


def extract_pattern_from_path(pattern, path: Path):
    if (m := re.match(pattern, Path(path).name)) is not None:
        info = m.groupdict()
    else:
        raise RuntimeError(f"Filename {path.name} does not match the provided pattern.")

    return info


def extract_metadata_from_bids(
    segmentation_path: Path,
    mri_data_path: Path,
) -> dict:
    """Extract subject, session, mri data type and segmentation name from filepath. \
        Assumes that naming follows the BIDS convention

    Args:
        segmentation_path (Path): Path so segmentation file
        mri_data_path (Path): Path to mri data file

    Raises:
        RuntimeError: If subject ID in the segmentation filename does not match the subject ID in the mri data filename

    Returns:
        dict: Combined subject, session, mri data type and segmentation name
    """

    seg_pattern = r"sub-(?P<subject>[^\.]+)_seg-(?P<segmentation>[^\.]+)"
    # Identify subject and segmentation from segmentation filename
    seg_info = extract_pattern_from_path(pattern=seg_pattern, path=segmentation_path)

    mri_data_pattern = r"sub-(?P<subject>[^\.]+)_(?P<session>ses-\d{2})_(?P<mri_data>[^\.]+)"
    # Identify subject, session and mri data type from mri data filename
    mri_info = extract_pattern_from_path(pattern=mri_data_pattern, path=mri_data_path)

    if mri_info["subject"] != seg_info["subject"]:
        raise RuntimeError(
            f"Subject ID mismatch between segmentation and MRI data: {seg_info['subject']} vs {mri_info['subject']}"
        )

    return seg_info | mri_info
