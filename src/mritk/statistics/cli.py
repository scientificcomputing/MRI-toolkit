import argparse
import typing
from pathlib import Path

import pandas as pd

from ..data import MRIData
from ..segmentation import Segmentation
from .compute_stats import generate_stats_dataframe_rois
from .metadata import extract_metadata_from_bids


def compute_mri_stats(
    segmentation: Path,
    mri: list[Path],
    output: Path,
    lut: Path | None = None,
    info: str | None = None,
    use_bids_metadata: bool = False,
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

    seg = Segmentation.from_file(segmentation)
    # Validate all MRI paths before starting
    for path in mri:
        if not path.exists():
            console.print(f"[bold red]Error:[/bold red] Missing MRI file: {path}")
            sys.exit(1)

    dataframes = []

    # Loop through MRI paths
    console.print("[bold green]Processing MRIs...[/bold green]")
    for i, path in enumerate(mri):
        if use_bids_metadata:
            try:
                bids_metadata = extract_metadata_from_bids(segmentation, path)
            except Exception as e:
                console.print(f"[bold red]Error extracting BIDS metadata:[/bold red] {e}")
                sys.exit(1)

            info_dict = (info_dict if info_dict else {}) | bids_metadata

        mri_object = MRIData.from_file(path)  # Load MRI data
        try:
            # Call the logic function
            # TODO: Add option to specify statistics to compute
            df = generate_stats_dataframe_rois(
                seg=seg,
                mri=mri_object,
                metadata=info_dict,
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


def get_stats_value(stats_file: Path, ROI: int, statistic: str, **kwargs):
    """
    Replaces the @click.command('get') decorated function.
    """
    import sys

    from rich.console import Console

    # Setup Rich
    console = Console()

    # Verify that csv exists
    if not stats_file.exists():
        console.print(f"[bold red]Error:[/bold red] Stats file not found: {stats_file}")
        sys.exit(1)

    # Process
    try:
        # Read csv
        df = pd.read_csv(stats_file, sep=";")

        # Verify that the requested statistic exists in the dataframe
        valid_statistics = set(df["statistic"])
        if statistic not in valid_statistics:
            console.print(
                f"[bold red]Error:[/bold red] Statistic '{statistic}' is invalid. Choose from: {', '.join(valid_statistics)}"
            )
            sys.exit(1)

        # Verify that the requested ROI exists in the dataframe
        valid_rois = set(df["ROI"])
        if ROI not in valid_rois:
            console.print(
                f"[bold red]Error:[/bold red] ROI '{ROI}' not found in stats file. Valid ROIs: {', '.join(map(str, valid_rois))}"
            )
            sys.exit(1)

        statistic_value = df.loc[(df["ROI"] == ROI) & (df["statistic"] == statistic), "value"]

        # Output
        console.print(
            f"[bold cyan]{statistic}[/bold cyan] for ROI \
                [bold green]{ROI}[/bold green] = [bold white]{statistic_value.item()}[/bold white]"
        )
        return statistic_value.item()  # Return as scalar

    except Exception as e:
        console.print(f"[bold red]Error reading stats file:[/bold red] {e}")
        sys.exit(1)


def add_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="stats-command", help="Available commands")

    # --- Compute Command ---
    parser_compute = subparsers.add_parser("compute", help="Compute MRI statistics", formatter_class=parser.formatter_class)
    parser_compute.add_argument(
        "--segmentation",
        "-s",
        type=Path,
        required=True,
        help="Path to segmentation file",
    )
    parser_compute.add_argument(
        "--mri",
        "-m",
        type=Path,
        nargs="+",
        required=True,
        help="Path to MRI data file(s)",
    )
    parser_compute.add_argument("--output", "-o", type=Path, required=True, help="Output CSV file path")
    parser_compute.add_argument("--lut", "-lt", dest="lut", type=Path, help="Path to Lookup Table")
    parser_compute.add_argument(
        "--info",
        "-i",
        type=str,
        help="Info dictionary as JSON string. \
            If using --use_bids_metadata, overlapping fields will be overwritten by BIDS metadata extraction.",
    )
    parser_compute.add_argument(
        "--use_bids_metadata",
        "-b",
        action="store_true",
        help="Assumes file naming follows BIDS convention and extracts metadata accordingly.\
            Checks that subject IDs match between segmentation and MRI data.",
    )
    parser_compute.set_defaults(func=compute_mri_stats)

    # --- Get Command ---
    parser_get = subparsers.add_parser("get", help="Get specific stats value", formatter_class=parser.formatter_class)
    parser_get.add_argument("--stats_file", "-f", type=Path, required=True, help="Path to stats CSV file")
    parser_get.add_argument("--ROI", "-r", type=int, required=True, help="Region of interest to extract")
    parser_get.add_argument(
        "--statistic",
        "-s",
        type=str,
        required=True,
        help="Statistic to retrieve (mean, std, etc.)",
    )
    parser_get.set_defaults(func=get_stats_value)


def dispatch(args: dict[str, typing.Any]):
    command = args.pop("stats-command")
    if command == "compute":
        compute_mri_stats(**args)
    elif command == "get":
        get_stats_value(**args)
    else:
        raise ValueError(f"Unknown command: {command}")
