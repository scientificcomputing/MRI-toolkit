# MRI Statistics Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tqdm.rich

from ..data.io import load_mri_data
from ..data.orientation import assert_same_space
from ..segmentation.groups import default_segmentation_groups
from ..segmentation.lookup_table import read_lut
from .utils import voxel_count_to_ml_scale, find_timestamp, prepend_info


def extract_metadata(
    file_path: Path,
    pattern: str | None = None,
    info_dict: dict[str, str] | None = None,
    required_keys: list[str] | None = None,
) -> dict:
    """
    Extracts metadata from a filename using a regex pattern, falling back to a dictionary.

    Args:
        file_path (Path): The path to the file.
        pattern (str, optional): Regex pattern with named capture groups.
        info_dict (dict, optional): Fallback dictionary if pattern is not provided.
        required_keys (list[str], optional): Keys to initialize with None if neither match.

    Returns:
        dict: A dictionary of the extracted metadata.

    Raises:
        RuntimeError: If a pattern is provided but the filename does not match.
    """
    if pattern is not None:
        if (m := re.match(rf"{pattern}", file_path.name)) is not None:
            return m.groupdict()
        else:
            raise RuntimeError(f"Filename {file_path.name} does not match the provided pattern.")

    required_keys = required_keys or []
    if info_dict is not None:
        return {k: info_dict.get(k) for k in required_keys}

    return {k: None for k in required_keys}


def get_regions_dictionary(seg_data: np.ndarray, lut_path: Optional[Path] = None) -> dict[str, list[int]]:
    """
    Builds a dictionary mapping region descriptions to their corresponding segmentation labels.

    Args:
        seg_data (np.ndarray): The segmentation array.
        lut_path (Path, optional): Path to the FreeSurfer Color Look-Up Table.

    Returns:
        dict[str, list[int]]: Mapping of region names to a list of label integers.
    """
    lut = read_lut(lut_path)
    seg_labels = np.unique(seg_data[seg_data != 0])

    lut_regions = lut.loc[lut.label.isin(seg_labels), ["label", "description"]].to_dict("records")

    regions = {
        **{d["description"]: sorted([d["label"]]) for d in lut_regions},
        **default_segmentation_groups(),
    }
    return regions


def compute_region_statistics(
    region_data: np.ndarray,
    labels: list[int],
    description: str,
    volscale: float,
    voxelcount: int,
) -> dict:
    """
    Computes statistical metrics (mean, std, percentiles, etc.) for a specific masked region.

    Args:
        region_data (np.ndarray): The raw MRI data values mapped to this region (includes NaNs).
        labels (list[int]): The segmentation label indices representing this region.
        description (str): Human-readable name of the region.
        volscale (float): Multiplier to convert voxel counts to milliliters.
        voxelcount (int): Total number of voxels in the region.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    record = {
        "label": ",".join([str(x) for x in labels]),
        "description": description,
        "voxelcount": voxelcount,
        "volume_ml": volscale * voxelcount,
    }

    if voxelcount == 0:
        return record

    num_nan = int((~np.isfinite(region_data)).sum())
    record["num_nan_values"] = num_nan

    if num_nan == voxelcount:
        return record

    # Filter out NaNs for the mathematical stats
    valid_data = region_data[np.isfinite(region_data)]

    stats = {
        "sum": float(np.sum(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "min": float(np.min(valid_data)),
        **{f"PC{pc}": float(np.quantile(valid_data, pc / 100)) for pc in [1, 5, 25, 75, 90, 95, 99]},
        "max": float(np.max(valid_data)),
    }

    return {**record, **stats}


def generate_stats_dataframe(
    seg_path: Path,
    mri_path: Path,
    timestamp_path: str | Path | None = None,
    timestamp_sequence: str | Path | None = None,
    seg_pattern: str | None = None,
    mri_data_pattern: str | None = None,
    lut_path: Path | None = None,
    info_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame containing descriptive statistics of MRI data grouped by segmentation regions.

    Args:
        seg_path (Path): Path to the segmentation NIfTI file.
        mri_path (Path): Path to the underlying MRI data NIfTI file.
        timestamp_path (str | Path, optional): Path to the timetable TSV file.
        timestamp_sequence (str | Path, optional): Sequence label to query in the timetable.
        seg_pattern (str, optional): Regex to extract metadata from the seg_path filename.
        mri_data_pattern (str, optional): Regex to extract metadata from the mri_path filename.
        lut_path (Path, optional): Path to the look-up table.
        info_dict (dict, optional): Fallback dictionary for metadata.

    Returns:
        pd.DataFrame: A formatted DataFrame with statistics for all identified regions.
    """
    # Load and validate the data
    mri = load_mri_data(mri_path, dtype=np.single)
    seg = load_mri_data(seg_path, dtype=np.int16)
    assert_same_space(seg, mri)

    # Resolve metadata
    seg_info = extract_metadata(seg_path, seg_pattern, info_dict, ["segmentation", "subject"])
    mri_info = extract_metadata(mri_path, mri_data_pattern, info_dict, ["mri_data", "subject", "session"])
    info = seg_info | mri_info

    # Resolve timestamps
    info["timestamp"] = None
    if timestamp_path is not None:
        try:
            info["timestamp"] = find_timestamp(
                Path(str(timestamp_path)),
                str(timestamp_sequence),
                str(info.get("subject")),
                str(info.get("session")),
            )
        except (ValueError, RuntimeError, KeyError):
            pass

    regions = get_regions_dictionary(seg.data, lut_path)
    volscale = voxel_count_to_ml_scale(seg.affine)
    records = []

    # Iterate over regions and compute stats
    for description, labels in tqdm.rich.tqdm(regions.items(), total=len(regions)):
        region_mask = np.isin(seg.data, labels)
        voxelcount = region_mask.sum()

        # Extract raw data for this region (including NaNs)
        region_data = mri.data[region_mask]

        record = compute_region_statistics(
            region_data=region_data,
            labels=labels,
            description=description,
            volscale=volscale,
            voxelcount=voxelcount,
        )
        records.append(record)

    # Format output
    dframe = pd.DataFrame.from_records(records)
    dframe = prepend_info(
        dframe,
        segmentation=info.get("segmentation"),
        mri_data=info.get("mri_data"),
        subject=info.get("subject"),
        session=info.get("session"),
        timestamp=info.get("timestamp"),
    )
    return dframe
