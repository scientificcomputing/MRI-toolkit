"""MRI-toolkit provides a set of features dedicated to MRI data post-processing and analysis."""

import argparse
import logging
from importlib.metadata import metadata
from pathlib import Path
from typing import Optional, Sequence

from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter

from . import concentration, datasets, hybrid, info, looklocker, mixed, napari, r1, show, statistics


def version_info():
    import sys

    import nibabel as nib
    import numpy as np
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    meta = metadata("mritk")
    toolkit_version = meta["Version"]
    python_version = sys.version.split()[0]

    table = Table(
        title="MRI Toolkit Environment",
        box=box.ROUNDED,  # Nice rounded corners
        show_lines=True,  # Separator lines between rows
        header_style="bold magenta",
    )

    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Version", style="green", justify="right")

    table.add_row("mri-toolkit", toolkit_version)
    table.add_row("Python", python_version)
    table.add_row("Nibabel", nib.__version__)
    table.add_row("Numpy", np.__version__)

    console.print(table)


def add_extra_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--no-rich", action="store_true", help="Disable rich logging and use standard console output.")
    parser.add_argument("--logfile", type=Path, help="Path to a log file to save logs (optional).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RichHelpFormatter)
    parser.add_argument("--version", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    # Download test data parser
    datasets_parser = subparsers.add_parser("datasets", help="Download datasets", formatter_class=parser.formatter_class)
    datasets.add_arguments(datasets_parser, extra_args_cb=add_extra_arguments)

    info_parser = subparsers.add_parser("info", help="Display information about a file", formatter_class=parser.formatter_class)
    info_parser.add_argument("file", type=Path, help="File to display information about")

    info_parser.add_argument("--json", action="store_true", help="Output information in JSON format")

    stats_parser = subparsers.add_parser("stats", help="Compute MRI statistics", formatter_class=parser.formatter_class)
    statistics.cli.add_arguments(stats_parser)

    show_parser = subparsers.add_parser("show", help="Show MRI data in a terminal", formatter_class=parser.formatter_class)
    show.add_arguments(show_parser)

    napari_parser = subparsers.add_parser("napari", help="Show MRI data using napari", formatter_class=parser.formatter_class)
    napari.add_arguments(napari_parser)

    looklocker_parser = subparsers.add_parser(
        "looklocker", help="Process Look-Locker data", formatter_class=parser.formatter_class
    )
    looklocker.add_arguments(looklocker_parser, extra_args_cb=add_extra_arguments)

    hybrid_parser = subparsers.add_parser(
        "hybrid", help="Generate a hybrid T1 map by merging Look-Locker and Mixed maps.", formatter_class=parser.formatter_class
    )
    hybrid.add_arguments(hybrid_parser, extra_args_cb=add_extra_arguments)

    mixed_parser = subparsers.add_parser(
        "mixed", help="Generate a Mixed T1 map from Look-Locker data.", formatter_class=parser.formatter_class
    )
    mixed.add_arguments(mixed_parser, extra_args_cb=add_extra_arguments)

    t1_to_r1_parser = subparsers.add_parser(
        "t1-to-r1", help="Convert a T1 map to an R1 map.", formatter_class=parser.formatter_class
    )
    r1.add_arguments(t1_to_r1_parser, extra_args_cb=add_extra_arguments)

    concentration_parser = subparsers.add_parser(
        "concentration", help="Compute concentration maps.", formatter_class=parser.formatter_class
    )
    concentration.add_arguments(concentration_parser, extra_args_cb=add_extra_arguments)

    return parser


def dispatch(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))

    if args.pop("version"):
        version_info()
        return 0
    verbose = args.pop("verbose", False)
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    no_rich = args.pop("no_rich", False)
    handlers: list[logging.Handler] = [logging.StreamHandler()] if no_rich else [RichHandler()]

    logfile = args.pop("logfile", None)
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    command = args.pop("command")
    logger = logging.getLogger(__name__)
    try:
        if command == "datasets":
            datasets.dispatch(args)
        elif command == "info":
            file = args.pop("file")
            info.nifty_info(file, json_output=args.pop("json"))
        elif command == "stats":
            statistics.cli.dispatch(args)
        elif command == "show":
            show.dispatch(args)
        elif command == "napari":
            napari.dispatch(args)
        elif command == "looklocker":
            looklocker.dispatch(args)
        elif command == "hybrid":
            hybrid.dispatch(args)
        elif command == "mixed":
            mixed.dispatch(args)
        elif command == "t1-to-r1":
            r1.dispatch(args)
        elif command == "concentration":
            concentration.dispatch(args)

        else:
            logger.error(f"Unknown command {command}")
            parser.print_help()
    except ValueError as e:
        logger.error(e, exc_info=True, stacklevel=2)
        parser.print_help()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = setup_parser()
    return dispatch(parser, argv)
