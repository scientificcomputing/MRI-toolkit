import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

# Assuming relative imports based on your previous file structure
from .data.io import load_mri_data


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("file", type=Path, help="File to show")
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colormap to use for displaying the image (default: gray)",
    )
    parser.add_argument(
        "--slice-x",
        type=float,
        default=0.5,
        help="Relative position (0-1) of the sagittal slice to display (default: 0.5)",
    )
    parser.add_argument(
        "--slice-y",
        type=float,
        default=0.5,
        help="Relative position (0-1) of the coronal slice to display (default: 0.5)",
    )
    parser.add_argument(
        "--slice-z",
        type=float,
        default=0.5,
        help="Relative position (0-1) of the axial slice to display (default: 0.5)",
    )


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
        from textual_image.renderable import Image as TermImage
        import PIL.Image
    except ImportError:
        console = Console()
        console.print(
            "[bold red]Error:[/bold red] The 'textual_image' and 'pillow' "
            "packages are required to use the 'show' command. "
            "Please install with: 'pip install mri-toolkit[show]'"
        )
        return

    # 1. Load Data
    # Assuming args is a dict or Namespace. Adapting to your snippet's usage:

    file_path = args.pop("file")
    cmap_name = args.pop("cmap", "gray")
    slize_x = args.pop("slice_x", 0.5)
    slize_y = args.pop("slice_y", 0.5)
    slize_z = args.pop("slice_z", 0.5)

    console = Console()
    console.print(f"[bold green]Loading MRI data from:[/bold green] {file_path}")

    mri_resource = load_mri_data(file_path)
    data = mri_resource.data

    # 2. Define Slice Indices (Middle of the brain)
    x_idx = int(data.shape[0] * slize_x)
    y_idx = int(data.shape[1] * slize_y)
    z_idx = int(data.shape[2] * slize_z)

    # 3. Extract Slices
    # orientation in load_mri_data is typically RAS (Right, Anterior, Superior)
    # Numpy origin is top-left. We often need to rotate/flip for correct medical view.

    # Sagittal View (Side): Fix X. Axes are Y (Ant) and Z (Sup).
    # We rotate 90 deg so Superior is "Up".
    slice_sagittal = np.rot90(data[x_idx, :, :])

    # Coronal View (Front): Fix Y. Axes are X (Right) and Z (Sup).
    slice_coronal = np.rot90(data[:, y_idx, :])

    # Axial View (Top-down): Fix Z. Axes are X (Right) and Y (Ant).
    slice_axial = np.rot90(data[:, :, z_idx])

    # 4. Prepare Images
    slices = [("Sagittal", slice_sagittal), ("Coronal", slice_coronal), ("Axial", slice_axial)]

    panels = []

    try:
        from matplotlib import cm
    except ImportError:
        cmap = lambda x: x / 255  # Identity if matplotlib not available
    else:
        cmap = cm.get_cmap(cmap_name)

    for title, slice_data in slices:
        # Normalize data to 0-255
        img_uint8 = normalize_to_uint8(slice_data)

        # Create PIL Image
        # pil_image = PIL.Image.fromarray(img_uint8)
        pil_image = PIL.Image.fromarray((cmap(img_uint8) * 255).astype(np.uint8))

        # Create Terminal Image
        term_img = TermImage(pil_image)

        # Add to list as a Panel
        panels.append(Panel(term_img, title=title, expand=False))

    # 5. Display
    console.print(Columns(panels, equal=True))
