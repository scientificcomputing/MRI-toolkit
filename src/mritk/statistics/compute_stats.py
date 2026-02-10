"""MRI Statistics Module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
from typing import Optional
import re
import numpy as np
import pandas as pd
import tqdm.rich

from ..data.io import load_mri_data
from ..data.orientation import assert_same_space
from ..segmentation.groups import default_segmentation_groups
from ..segmentation.lookup_table import read_lut
from .utils import voxel_count_to_ml_scale, find_timestamp, prepend_info


def generate_stats_dataframe(
    seg_path: Path,
    mri_path: Path,
    timestamp_path: Optional[str | Path] = None,
    timestamp_sequence: Optional[str | Path] = None,
    seg_pattern: Optional[str | Path] = None,
    mri_data_pattern: Optional[str | Path] = None,
    lut_path: Optional[Path] = None,
    info_dict: Optional[dict] = None,
) -> pd.DataFrame:
    # Load the data (mri and seg)
    mri = load_mri_data(mri_path, dtype=np.single)
    seg = load_mri_data(seg_path, dtype=np.int16)
    assert_same_space(seg, mri)
    # Load LUT
    lut = read_lut(lut_path)
    # Get LUT info
    seg_labels = np.unique(seg.data[seg.data != 0])
    lut_regions = lut.loc[lut.label.isin(seg_labels), ["label", "description"]].to_dict("records")
    regions = {
        **{d["description"]: sorted([d["label"]]) for d in lut_regions},
        **default_segmentation_groups(),
    }
    # Get SEG info
    seg_info = {}
    if seg_pattern is not None:
        seg_pattern = rf"{seg_pattern}"
        if (m := re.match(seg_pattern, Path(seg_path).name)) is not None:
            seg_info = m.groupdict()
        else:
            raise RuntimeError(f"Segmentation filename {seg_path.name} does not match the provided pattern.")
    elif info_dict is not None:
        seg_info["segmentation"] = info_dict["segmentation"] if "segmentation" in info_dict else None
        seg_info["subject"] = info_dict["subject"] if "subject" in info_dict else None
    else:
        seg_info = {"segmentation": None, "subject": None}
    # Get MRI info
    mri_info = {}
    if mri_data_pattern is not None:
        mri_data_pattern = rf"{mri_data_pattern}"
        if (m := re.match(mri_data_pattern, Path(mri_path).name)) is not None:
            mri_info = m.groupdict()
        else:
            raise RuntimeError(f"MRI data filename {mri_path.name} does not match the provided pattern.")
    elif info_dict is not None:
        mri_info["mri_data"] = info_dict["mri_data"] if "mri_data" in info_dict else None
        mri_info["subject"] = info_dict["subject"] if "subject" in info_dict else None
        mri_info["session"] = info_dict["session"] if "session" in info_dict else None
    else:
        mri_info = {"mri_data": None, "subject": None, "session": None}
    # Get timestamp
    if timestamp_path is not None:
        try:
            mri_info["timestamp"] = find_timestamp(
                Path(str(timestamp_path)),
                str(timestamp_sequence),
                str(mri_info["subject"]),
                str(mri_info["session"]),
            )
        except (ValueError, RuntimeError, KeyError):
            mri_info["timestamp"] = None
    else:
        mri_info["timestamp"] = None

    info = seg_info | mri_info

    records = []
    finite_mask = np.isfinite(mri.data)
    volscale = voxel_count_to_ml_scale(seg.affine)

    for description, labels in tqdm.rich.tqdm(regions.items(), total=len(regions)):
        region_mask = np.isin(seg.data, labels)
        voxelcount = region_mask.sum()
        record = {
            "label": ",".join([str(x) for x in labels]),
            "description": description,
            "voxelcount": voxelcount,
            "volume_ml": volscale * voxelcount,
        }
        if voxelcount == 0:
            records.append(record)
            continue

        data_mask = region_mask * finite_mask
        region_data = mri.data[data_mask]
        num_nan = (~np.isfinite(region_data)).sum()
        record["num_nan_values"] = num_nan
        if num_nan == voxelcount:
            records.append(record)
            continue

        stats = {
            "sum": np.sum(region_data),
            "mean": np.mean(region_data),
            "median": np.median(region_data),
            "std": np.std(region_data),
            "min": np.min(region_data),
            **{f"PC{pc}": np.quantile(region_data, pc / 100) for pc in [1, 5, 25, 75, 90, 95, 99]},
            "max": np.max(region_data),
        }
        records.append({**record, **stats})

    dframe = pd.DataFrame.from_records(records)
    dframe = prepend_info(
        dframe,
        segmentation=info["segmentation"],
        mri_data=info["mri_data"],
        subject=info["subject"],
        session=info["session"],
        timestamp=info["timestamp"],
    )
    return dframe
