# MRI Segmentation - Lookup Table (LUT) Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import re
import os
from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve


def read_lut(filename: Path | str | None) -> pd.DataFrame:
    if (filename is None) or (not Path(filename).exists()):
        filename = "FreeSurferColorLUT.txt"
        if "FREESURFER_HOME" in os.environ:
            filedir = Path(os.environ["FREESURFER_HOME"])
        else:
            filedir = Path.cwd()
        filename = filedir / filename
        if not filename.exists():
            url = "https://github.com/freesurfer/freesurfer/raw/dev/distribution/FreeSurferColorLUT.txt"
            urlretrieve(url, filename)
    lut_regex = re.compile(
        r"^(?P<label>\d+)\s+(?P<description>[_\da-zA-Z-]+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+(?P<A>\d+)"
    )
    with open(filename, "r") as f:
        records = [lut_record(m) for m in map(lut_regex.match, f) if m is not None]
    return pd.DataFrame.from_records(records)


def write_lut(filename: Path | str, table: pd.DataFrame):
    newtable = table.copy()
    for col in "RGB":
        newtable[col] = (newtable[col] * 255).astype(int)
    newtable["A"] = 255 - (newtable["A"] * 255).astype(int)
    newtable.to_csv(filename, sep="\t", index=False, header=False)


def lut_record(match: re.Match) -> dict[str, str | float]:
    groupdict = match.groupdict()
    return {
        "label": int(groupdict["label"]),
        "description": groupdict["description"],
        "R": float(groupdict["R"]) / 255,
        "G": float(groupdict["G"]) / 255,
        "B": float(groupdict["B"]) / 255,
        "A": 1.0 - float(groupdict["A"]) / 255,
    }
