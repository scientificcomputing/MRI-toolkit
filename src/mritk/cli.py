import logging
from pathlib import Path
import argparse
from typing import Sequence, Optional

from . import download_data


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest="command")

    # Download test data parser
    download_parser = subparsers.add_parser("download-test-data", help="Download test data")
    download_parser.add_argument("outdir", type=Path, help="Output directory to download test data")

    return parser


def dispatch(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    command = args.pop("command")
    logger = logging.getLogger(__name__)
    try:
        if command == "download-test-data":
            outdir = args.pop("outdir")
            download_data.download_test_data(outdir)
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
