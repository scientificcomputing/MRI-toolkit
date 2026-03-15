# MRI Statistics Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm.rich

from .data import MRIData
from .segmentation import default_segmentation_groups, read_lut
from .testing import assert_same_space


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


def get_regions_dictionary(seg_data: np.ndarray, lut_path: Path | None = None) -> dict[str, list[int]]:
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
    mri = MRIData.from_file(mri_path, dtype=np.single)
    seg = MRIData.from_file(seg_path, dtype=np.int16)
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


def compute_mri_stats(
    segmentation: Path,
    mri: list[Path],
    output: Path,
    timetable: Path | None = None,
    timelabel: str | None = None,
    seg_regex: str | None = None,
    mri_regex: str | None = None,
    lut: Path | None = None,
    info: str | None = None,
    **kwargs,
):
    import json
    import sys

    from rich.console import Console
    from rich.panel import Panel

    # Setup Rich
    console = Console()

    # Parse info dict from JSON string if provided
    info_dict = None
    if info:
        try:
            info_dict = json.loads(info)
        except json.JSONDecodeError:
            console.print("[bold red]Error:[/bold red] --info must be a valid JSON string.")
            sys.exit(1)

    if not segmentation.exists():
        console.print(f"[bold red]Error:[/bold red] Missing segmentation file: {segmentation}")
        sys.exit(1)

    # Validate all MRI paths before starting
    for path in mri:
        if not path.exists():
            console.print(f"[bold red]Error:[/bold red] Missing MRI file: {path}")
            sys.exit(1)

    dataframes = []

    # Loop through MRI paths
    console.print("[bold green]Processing MRIs...[/bold green]")
    for i, path in enumerate(mri):
        # console.print(f"[blue]Processing MRI {i + 1}/{len(mri)}:[/blue] {path.name}")

        try:
            # Call the logic function
            df = generate_stats_dataframe(
                seg_path=segmentation,
                mri_path=path,
                timestamp_path=timetable,
                timestamp_sequence=timelabel,
                seg_pattern=seg_regex,
                mri_data_pattern=mri_regex,
                lut_path=lut,
                info_dict=info_dict,
            )
            dataframes.append(df)
        except Exception as e:
            console.print(f"[bold red]Failed to process {path.name}:[/bold red] {e}")
            sys.exit(1)

    if dataframes:
        final_df = pd.concat(dataframes)
        final_df.to_csv(output, sep=";", index=False)
        console.print(
            Panel(
                f"Stats successfully saved to:\n[bold green]{output}[/bold green]",
                title="Success",
                expand=False,
            )
        )
    else:
        console.print("[yellow]No dataframes generated.[/yellow]")


def get_stats_value(stats_file: Path, region: str, info: str, **kwargs):
    import sys

    from rich.console import Console

    # Setup Rich
    console = Console()

    # Validate inputs
    valid_regions = default_segmentation_groups().keys()
    if region not in valid_regions:
        console.print(f"[bold red]Error:[/bold red] Region '{region}' not found in default segmentation groups.")
        sys.exit(1)

    valid_infos = [
        "sum",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "PC1",
        "PC5",
        "PC25",
        "PC75",
        "PC90",
        "PC95",
        "PC99",
    ]
    if info not in valid_infos:
        console.print(f"[bold red]Error:[/bold red] Info '{info}' is invalid. Choose from: {', '.join(valid_infos)}")
        sys.exit(1)

    if not stats_file.exists():
        console.print(f"[bold red]Error:[/bold red] Stats file not found: {stats_file}")
        sys.exit(1)

    # Process
    try:
        df = pd.read_csv(stats_file, sep=";")
        region_row = df.loc[df["description"] == region]

        if region_row.empty:
            console.print(f"[red]Region '{region}' not found in the stats file.[/red]")
            sys.exit(1)

        info_value = region_row[info].values[0]

        # Output
        console.print(
            f"[bold cyan]{info}[/bold cyan] for [bold green]{region}[/bold green] = [bold white]{info_value}[/bold white]"
        )
        return info_value

    except Exception as e:
        console.print(f"[bold red]Error reading stats file:[/bold red] {e}")
        sys.exit(1)


def add_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="stats-command", help="Available commands")

    # --- Compute Command ---
    parser_compute = subparsers.add_parser("compute", help="Compute MRI statistics", formatter_class=parser.formatter_class)
    parser_compute.add_argument("--segmentation", "-s", type=Path, required=True, help="Path to segmentation file")
    parser_compute.add_argument("--mri", "-m", type=Path, nargs="+", required=True, help="Path to MRI data file(s)")
    parser_compute.add_argument("--output", "-o", type=Path, required=True, help="Output CSV file path")
    parser_compute.add_argument("--timetable", "-t", type=Path, help="Path to timetable file")
    parser_compute.add_argument("--timelabel", "-l", dest="timelabel", type=str, help="Time label sequence")
    parser_compute.add_argument(
        "--seg_regex",
        "-sr",
        dest="seg_regex",
        type=str,
        help="Regex pattern for segmentation filename",
    )
    parser_compute.add_argument("--mri_regex", "-mr", dest="mri_regex", type=str, help="Regex pattern for MRI filename")
    parser_compute.add_argument("--lut", "-lt", dest="lut", type=Path, help="Path to Lookup Table")
    parser_compute.add_argument("--info", "-i", type=str, help="Info dictionary as JSON string")
    parser_compute.set_defaults(func=compute_mri_stats)

    # --- Get Command ---
    parser_get = subparsers.add_parser("get", help="Get specific stats value", formatter_class=parser.formatter_class)
    parser_get.add_argument("--stats_file", "-f", type=Path, required=True, help="Path to stats CSV file")
    parser_get.add_argument("--region", "-r", type=str, required=True, help="Region description")
    parser_get.add_argument("--info", "-i", type=str, required=True, help="Statistic to retrieve (mean, std, etc.)")
    parser_get.set_defaults(func=get_stats_value)


def dispatch(args: dict[str, typing.Any]):
    command = args.pop("stats-command")
    if command == "compute":
        compute_mri_stats(**args)
    elif command == "get":
        get_stats_value(**args)
    else:
        raise ValueError(f"Unknown command: {command}")
