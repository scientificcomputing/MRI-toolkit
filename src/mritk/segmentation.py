# MRI Segmentation - Lookup Table (LUT) Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import logging
import os
import re
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger(__name__)


# Regex to match a standard FreeSurfer Color LUT record line
LUT_REGEX = re.compile(r"^(?P<label>\d+)\s+(?P<description>[_\da-zA-Z-]+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+(?P<A>\d+)")
CORPUS_CALLOSUM = [251, 252, 253, 254, 255]

CEREBRAL_WM_RANGES = [
    *[2, 41],  # aseg left/right cerebral white labels
    *list(range(3000, 3036)),  # wmparc-left-labels
    *list(range(4000, 4036)),  # wmparc-right-labels
    *[5001, 5002],
    *[28, 60],  # VentralDC included in white matter subdomain
    *CORPUS_CALLOSUM,  # Corpus callosum
    *[31, 63],  # Choroid plexus.
]

CEREBRAL_CGM_RANGES = [
    *[3, 42],  # aseg left/right cortical gm
    *list(range(1000, 1036)),  # aparc left labels
    *list(range(2000, 2036)),  # aparc right labels
]

SUBCORTICAL_GM_RANGES = [
    *(10, 49),  # Thalamus,
    *(11, 50),  # Caudate,
    *(12, 51),  # Putamen,
    *(13, 52),  # pallidum
    *(17, 53),  # hippocampus
    *(18, 54),  # amygdala
    *(26, 58),  # accumbens
]

VENTRICLES = [
    4,  # Left lateral ventricle
    5,  # Left inferior lateral ventricle
    14,  # Third ventricle
    15,  # Fourth ventricle
    43,  # Right lateral ventricle
    44,  # Right inferior lateral ventricle
]

FREESURFER_CSF = [
    *VENTRICLES,
    24,  # Generic CSF
]

CORTICAL_CSF = [
    *(x + 15000 for x in CEREBRAL_CGM_RANGES),
]

SEGMENTATION_GROUPS = {
    "cerebral-wm": CEREBRAL_WM_RANGES,
    "cerebral-cortex": CEREBRAL_CGM_RANGES,
    "cerebellar-wm": [7, 46],  # left, right
    "cerebellar-cortex": [8, 47],  # left, right
    "csf-freesurfer": FREESURFER_CSF,
    "cortical-csf": CORTICAL_CSF,
    "corpus-callosum": CORPUS_CALLOSUM,
    "subcortical-gm": SUBCORTICAL_GM_RANGES,
}


def default_segmentation_groups() -> dict[str, list[int]]:
    """
    Returns the default grouping of FreeSurfer labels into brain regions.

    Returns:
        dict[str, list[int]]: A dictionary mapping human-readable anatomical
        region names (e.g., 'cerebral-wm') to a list of integer labels
        corresponding to those regions in the FreeSurfer Color LUT.
    """
    return {**SEGMENTATION_GROUPS}


def lut_record(match: re.Match) -> dict[str, str | float | int]:
    """
    Parses a regular expression match of a LUT line into a formatted dictionary.

    Normalizes RGB values from [0, 255] to [0.0, 1.0].
    Inverts the Alpha channel (FreeSurfer uses A=0 for opaque and A=255 for transparent,
    whereas standard rendering uses A=1.0 for opaque and A=0.0 for transparent).

    Args:
        match (re.Match): A regex match object containing label, description, R, G, B, and A groups.

    Returns:
        dict[str, str | float | int]: A parsed dictionary representing the LUT record.
    """
    groupdict = match.groupdict()
    return {
        "label": int(groupdict["label"]),
        "description": groupdict["description"],
        "R": float(groupdict["R"]) / 255.0,
        "G": float(groupdict["G"]) / 255.0,
        "B": float(groupdict["B"]) / 255.0,
        "A": 1.0 - float(groupdict["A"]) / 255.0,
    }


def validate_lut_file(filepath: Path) -> bool:
    """
    Validates that a file exists and contains valid FreeSurfer LUT records.

    Instead of a strict checksum (which would break custom LUTs), this
    checks if the file contains at least one valid record line within
    its first 50 lines.

    Args:
        filepath (Path): Path to the file to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                if LUT_REGEX.match(line):
                    return True
    except (IOError, UnicodeDecodeError):
        return False

    return False


def resolve_lut_path(filename: Path | None = None) -> Path:
    """
    Resolves and validates the file path for a Color Lookup Table.

    If a filename is provided, it validates it. If it doesn't exist, it uses
    that path as the download target. If no filename is provided, it defaults
    to 'FreeSurferColorLUT.txt' in FREESURFER_HOME or the current directory.
    Missing or invalid files trigger a fresh download from the official repository.

    Args:
        filename (Path | None, optional): Requested path to the LUT file. Defaults to None.

    Returns:
        Path: The resolved and validated absolute path to the LUT file.

    Raises:
        ValueError: If a provided file exists but has invalid formatting.
        RuntimeError: If the downloaded fallback file is invalid.
    """
    if filename is not None:
        target_path = Path(filename)
        if target_path.exists():
            if validate_lut_file(target_path):
                return target_path
            else:
                raise ValueError(f"Provided LUT file '{target_path}' is invalid or corrupted.")
    else:
        # Default fallback target if none provided
        target_name = "FreeSurferColorLUT.txt"
        filedir = Path(os.environ.get("FREESURFER_HOME", Path.cwd()))
        target_path = filedir / target_name

    # If the target path doesn't exist (whether custom or default) or is invalid, download it.
    if not target_path.exists() or not validate_lut_file(target_path):
        logger.info(f"Valid LUT file not found. Downloading to {target_path}...")

        # Ensure the parent directory exists before downloading
        target_path.parent.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/freesurfer/freesurfer/raw/dev/distribution/FreeSurferColorLUT.txt"
        urlretrieve(url, target_path)

        if not validate_lut_file(target_path):
            raise RuntimeError(f"Downloaded LUT file at '{target_path}' is invalid. Download may have failed.")

    return target_path


def read_lut(filename: Path | None = None) -> pd.DataFrame:
    """
    Reads a FreeSurfer Color Lookup Table text file into a Pandas DataFrame.

    Missing files will be automatically downloaded. RGBA values are normalized to floats [0.0, 1.0].

    Args:
        filename (Path | None, optional): Path to the LUT text file. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns ['label', 'description', 'R', 'G', 'B', 'A'].
    """
    resolved_path = resolve_lut_path(filename)

    with open(resolved_path, "r", encoding="utf-8") as f:
        records = [lut_record(m) for m in map(LUT_REGEX.match, f) if m is not None]

    return pd.DataFrame.from_records(records)


def write_lut(filename: Path, table: pd.DataFrame):
    """
    Writes a Pandas DataFrame back to the FreeSurfer Color Lookup Table text format.

    Reverses the normalization applied during `read_lut`, converting float [0.0, 1.0] RGBA
    values back to integer [0, 255] values, and re-inverting the Alpha channel.

    Args:
        filename (Path): Path where the LUT file will be saved.
        table (pd.DataFrame): The DataFrame containing the LUT records.
    """
    newtable = table.copy()

    # Re-scale RGB values to [0, 255]
    for col in ["R", "G", "B"]:
        newtable[col] = (newtable[col] * 255).astype(int)

    # Reverse Alpha inversion and scale to [0, 255]
    newtable["A"] = 255 - (newtable["A"] * 255).astype(int)

    # Save as tab-separated values without headers or indices
    newtable.to_csv(filename, sep="\t", index=False, header=False)
