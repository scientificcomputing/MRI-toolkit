import json
import typing
from pathlib import Path
import numpy as np
import nibabel as nib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def custom_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif np.isscalar(obj):
        return float(obj)
    else:
        return str(obj)


def nifty_info(filename: Path, json_output: bool = False) -> dict[str, typing.Any]:
    console = Console()

    # 1. Load the NIfTI file

    img = nib.load(str(filename))
    header = img.header
    affine = img.affine

    # --- Part A: Extracting Dimensions & Resolution ---
    img_shape = img.shape
    zooms = header.get_zooms()
    data_type = header.get_data_dtype()

    data = {
        "filename": str(filename),
        "shape": img_shape,
        "voxel_size_mm": zooms,
        "data_type": data_type,
        "affine": affine,
    }

    if json_output:
        print(json.dumps(data, default=custom_json, indent=4))
        return data

    # Create a nice header panel
    console.print(Panel(f"[bold blue]NIfTI File Analysis[/bold blue]\n[green]{filename}[/green]", expand=False))

    # Create a table for Basic Info
    info_table = Table(
        title="Basic Information",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    # Format the tuples/lists as strings for the table
    shape_str = ", ".join(map(str, img_shape))
    zoom_str = ", ".join([f"{z:.2f}" for z in zooms])

    info_table.add_row("Shape (x, y, z)", f"({shape_str})")
    info_table.add_row("Voxel Size (mm)", f"({zoom_str})")
    info_table.add_row("Data Type", str(data_type))

    console.print(info_table)

    # --- Part B: The Affine Matrix ---

    console.print("\n[bold]Affine Transformation Matrix[/bold] (Voxel → World)", style="yellow")

    # Create a specific table for the matrix to align numbers nicely
    matrix_table = Table(show_header=False, box=box.ROUNDED, border_style="dim")

    # Add 4 columns for the 4x4 matrix
    for _ in range(4):
        matrix_table.add_column(justify="right", style="green")

    for row in affine:
        # Format numbers to 4 decimal places for cleanliness
        matrix_table.add_row(*[f"{val: .4f}" for val in row])

    console.print(matrix_table)

    return data
