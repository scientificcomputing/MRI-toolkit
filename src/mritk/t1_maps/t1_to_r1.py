"""T1 to R1 module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
from pathlib import Path
from typing import Union

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data


def convert_T1_to_R1(
    T1map_mri: MRIData,
    scale: float = 1000,
    t1_low: float = 1,
    t1_high: float = float("inf"),
) -> MRIData:
    valid_t1 = (t1_low <= T1map_mri.data) * (T1map_mri.data <= t1_high)
    R1map = np.nan * np.zeros_like(T1map_mri.data)
    R1map[valid_t1] = scale / np.minimum(
        t1_high, np.maximum(t1_low, T1map_mri.data[valid_t1])
    )

    R1map_mri = MRIData(data=R1map, affine=T1map_mri.affine)
    return R1map_mri


def T1_to_R1(
    input_mri: Union[Path, MRIData],
    output: Path = None,
    scale: float = 1000,
    t1_low: float = 1,
    t1_high: float = float("inf"),
) -> MRIData:
    if isinstance(input_mri, Path):
        T1map_mri = load_mri_data(input_mri, dtype=np.single)
    elif isinstance(input_mri, MRIData):
        T1map_mri = input_mri
    else:
        raise ValueError(f"Input should be a Path or MRIData, got {type(input_mri)}")

    valid_t1 = (t1_low <= T1map_mri.data) * (T1map_mri.data <= t1_high)
    R1map = np.nan * np.zeros_like(T1map_mri.data)
    R1map[valid_t1] = scale / np.minimum(
        t1_high, np.maximum(t1_low, T1map_mri.data[valid_t1])
    )

    R1map_mri = MRIData(data=R1map, affine=T1map_mri.affine)

    if output is not None:
        save_mri_data(R1map_mri, output, dtype=np.single)
    return R1map_mri