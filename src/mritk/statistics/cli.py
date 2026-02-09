import argparse
import typing
from pathlib import Path
import pandas as pd

from ..segmentation.groups import default_segmentation_groups
from .compute_stats import generate_stats_dataframe


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
    import sys
    import json
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
    """
    Replaces the @click.command('get') decorated function.
    """
    import sys
    from rich.console import Console

    # Setup Rich
    console = Console()

    # Validate inputs
    valid_regions = default_segmentation_groups().keys()
    if region not in valid_regions:
        console.print(
            f"[bold red]Error:[/bold red] Region '{region}' "
            "not found in default segmentation groups."
        )
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
        console.print(
            f"[bold red]Error:[/bold red] Info '{info}' "
            f"is invalid. Choose from: {', '.join(valid_infos)}"
        )
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
            f"[bold cyan]{info}[/bold cyan] for [bold green]{region}[/bold green] "
            f"= [bold white]{info_value}[/bold white]"
        )
        return info_value

    except Exception as e:
        console.print(f"[bold red]Error reading stats file:[/bold red] {e}")
        sys.exit(1)


def add_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="stats-command", help="Available commands")

    # --- Compute Command ---
    parser_compute = subparsers.add_parser(
        "compute", help="Compute MRI statistics", formatter_class=parser.formatter_class
    )
    parser_compute.add_argument(
        "--segmentation", "-s", type=Path, required=True, help="Path to segmentation file"
    )
    parser_compute.add_argument(
        "--mri", "-m", type=Path, nargs="+", required=True, help="Path to MRI data file(s)"
    )
    parser_compute.add_argument(
        "--output", "-o", type=Path, required=True, help="Output CSV file path"
    )
    parser_compute.add_argument("--timetable", "-t", type=Path, help="Path to timetable file")
    parser_compute.add_argument(
        "--timelabel", "-l", dest="timelabel", type=str, help="Time label sequence"
    )
    parser_compute.add_argument(
        "--seg_regex",
        "-sr",
        dest="seg_regex",
        type=str,
        help="Regex pattern for segmentation filename",
    )
    parser_compute.add_argument(
        "--mri_regex", "-mr", dest="mri_regex", type=str, help="Regex pattern for MRI filename"
    )
    parser_compute.add_argument("--lut", "-lt", dest="lut", type=Path, help="Path to Lookup Table")
    parser_compute.add_argument("--info", "-i", type=str, help="Info dictionary as JSON string")
    parser_compute.set_defaults(func=compute_mri_stats)

    # --- Get Command ---
    parser_get = subparsers.add_parser(
        "get", help="Get specific stats value", formatter_class=parser.formatter_class
    )
    parser_get.add_argument(
        "--stats_file", "-f", type=Path, required=True, help="Path to stats CSV file"
    )
    parser_get.add_argument("--region", "-r", type=str, required=True, help="Region description")
    parser_get.add_argument(
        "--info", "-i", type=str, required=True, help="Statistic to retrieve (mean, std, etc.)"
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
