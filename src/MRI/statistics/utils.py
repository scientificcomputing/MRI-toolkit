"""MRI Statistics - Utils

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
from pathlib import Path
import pandas as pd


def voxel_count_to_ml_scale(affine: np.ndarray):
    return 1e-3 * np.linalg.det(affine[:3, :3])


def find_timestamp(
    timetable_path: Path,
    timestamp_sequence: str,
    subject: str,
    session: str,
) -> float:
    """Find single session timestamp"""
    try:
        timetable = pd.read_csv(timetable_path, sep="\t")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Timetable-file {timetable_path} is empty.")
    try:
        timestamp = timetable.loc[
            (timetable["sequence_label"].str.lower() == timestamp_sequence)
            & (timetable["subject"] == subject)
            & (timetable["session"] == session)
        ]["acquisition_relative_injection"]
    except ValueError as e:
        print(timetable)
        print(timestamp_sequence, subject)
        raise e
    return timestamp.item()


def prepend_info(df, **kwargs):
    nargs = len(kwargs)
    for key, val in kwargs.items():
        assert key not in df.columns, f"Column {key} already exist in df."
        df[key] = val
    return df[[*df.columns[-nargs:], *df.columns[:-nargs]]]
