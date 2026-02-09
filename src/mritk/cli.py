import logging
from importlib.metadata import metadata
from pathlib import Path
import argparse
from typing import Sequence, Optional

from rich_argparse import RichHelpFormatter

from . import download_data, info, statistics, show


def version_info():
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import sys
    import nibabel as nib
    import numpy as np

    console = Console()

    meta = metadata("mri-toolkit")
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


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("--version", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    # Download test data parser
    download_parser = subparsers.add_parser(
        "download-test-data", help="Download test data", formatter_class=parser.formatter_class
    )
    download_parser.add_argument("outdir", type=Path, help="Output directory to download test data")

    info_parser = subparsers.add_parser(
        "info", help="Display information about a file", formatter_class=parser.formatter_class
    )
    info_parser.add_argument("file", type=Path, help="File to display information about")

    info_parser.add_argument(
        "--json", action="store_true", help="Output information in JSON format"
    )

    stats_parser = subparsers.add_parser(
        "stats", help="Compute MRI statistics", formatter_class=parser.formatter_class
    )
    statistics.cli.add_arguments(stats_parser)

    show_parser = subparsers.add_parser(
        "show", help="Show MRI data in a terminal", formatter_class=parser.formatter_class
    )
    show.add_arguments(show_parser)

    return parser


def dispatch(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))

    if args.pop("version"):
        version_info()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    command = args.pop("command")
    logger = logging.getLogger(__name__)
    try:
        if command == "download-test-data":
            outdir = args.pop("outdir")
            download_data.download_test_data(outdir)
        elif command == "info":
            file = args.pop("file")
            info.nifty_info(file, json_output=args.pop("json"))
        elif command == "stats":
            statistics.cli.dispatch(args)
        elif command == "show":
            show.dispatch(args)
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
