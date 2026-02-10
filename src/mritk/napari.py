import argparse
from pathlib import Path

import numpy as np
from rich.console import Console

# Assuming relative imports based on your previous file structure
from .data.io import load_mri_data


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("files", nargs="+", type=Path, help="Files to show")


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize array values to 0-255 uint8 range for image display."""
    # Handle NaNs and Infs
    data = np.nan_to_num(data)

    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        # Linear scaling to 0-255
        normalized = (data - d_min) / (d_max - d_min) * 255
    else:
        normalized = np.zeros_like(data)

    return normalized.astype(np.uint8)


def dispatch(args):
    """
    Displays three orthogonal slices (Sagittal, Coronal, Axial) of an MRI file
    in the terminal.
    """
    try:
        import napari
    except ImportError:
        console = Console()
        console.print(
            "[bold red]Error:[/bold red] The 'napari' library is required to use the 'napari' command. "
            "Please install it with 'pip install mri-toolkit[napari]'"
        )
        return
    # 1. Load Data
    # Assuming args is a dict or Namespace. Adapting to your snippet's usage:

    file_paths = args.pop("files")

    viewer = napari.Viewer()
    for file_path in file_paths:
        console = Console()
        console.print(f"[bold green]Loading MRI data from:[/bold green] {file_path}")

        mri_resource = load_mri_data(file_path)
        data = mri_resource.data
        viewer.add_image(data, name=file_path.stem)

    napari.run()
