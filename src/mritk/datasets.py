"""MRI -- Download data for testing

Copyright (C) 2026   Henrik Finsberg (henriknf@simula.no)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import logging
from dataclasses import dataclass
import zipfile
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    name: str
    links: dict[str, str]
    description: str = ""
    doi: str = ""
    license: str = ""


def get_datasets() -> dict[str, Dataset]:
    return {
        "test-data": Dataset(
            name="Test Data",
            description="A small test dataset for testing functionality (based on the Gonzo dataset).",
            doi="10.5281/zenodo.14266867",
            license="CC-BY-4.0",
            links={
                "mri-processed.zip": "https://zenodo.org/records/14266867/files/mri-processed.zip?download=1",
                "timetable.tsv": "https://github.com/jorgenriseth/gonzo/blob/main/mri_dataset/timetable.tsv?raw=true",
            },
        ),
        "gonzo": Dataset(
            name="The Gonzo Dataset",
            description="""
            We present the Gonzo dataset: brain MRI and derivative data of one healthy-appearing male human volunteer
            before and during the 72 hours after injection of the contrast agent gadobutrol into the cerebrospinal
            fluid (CSF) of the spinal canal (intrathecal injection). The data records show the temporal and spatial
            evolution of the contrast agent in CSF, brain, and adjacent structures. The MRI data includes T1-weighted
            images, Look-Locker inversion recovery (LL, a technique to determine T1 values), a mixed inversion-recovery
            spin-echo sequence (Mixed) for all time points (one pre-contrast and four post-contrast acquisitions) and,
            in addition, T2-weighted, FLAIR, and dynamic DTI data for the pre-contrast session. In addition to raw data,
            we provide derivatives with the goal of allowing for numerical simulations of the studied tracer transport process.
            This includes T1 maps (from LL and Mixed) and tracer concentration maps, diffusion tensor maps, as well as
            unstructured triangulated volume meshes of the brain geometry and associated field data (MRI and derived data mapped
            onto the computational mesh). We provide brain region markers obtained with a FreeSurfer-based analysis pipeline.
            An initial regional statistical analysis of the data is presented. The data can be used to study the transport
            behaviour and the underlying processes of a tracer in the human brain. Tracer transport is both relevant to study
            water transport as well as new pathways for drug delivery. The composition of the data set allows both reuse
            by the image processing and the simulation science communities. The dataset is meant to contribute and
            inspire new studies into the understanding of transport processes in the brain and into method development
            regarding image analysis and simulation of transport processes.""",
            doi="10.5281/zenodo.14266867",
            license="CC-BY-4.0",
            links={
                "data-descriptor-preprint.pdf": "https://zenodo.org/records/14266867/files/data-descriptor-preprint.pdf?download=1",
                "fastsurfer.zip": "https://zenodo.org/records/14266867/files/fastsurfer.zip?download=1",
                "freesurfer.zip": "https://zenodo.org/records/14266867/files/freesurfer.zip?download=1",
                "mesh-data.zip": "https://zenodo.org/records/14266867/files/mesh-data.zip?download=1",
                "mri-dataset-precontrast-only.zip": "https://zenodo.org/records/14266867/files/mri-dataset-precontrast-only.zip?download=1",
                "mri-dataset.zip": "https://zenodo.org/records/14266867/files/mri-dataset.zip?download=1",
                "mri-processed.zip": "https://zenodo.org/records/14266867/files/mri-processed.zip?download=1",
                "README.md": "https://zenodo.org/records/14266867/files/README.md?download=1",
                "surfaces.zip": "https://zenodo.org/records/14266867/files/surfaces.zip?download=1",
            },
        ),
        "ratbrain": Dataset(
            name="Ratbrain Mesh",
            description="""
            This repository contains a collection of files that were used in the article Poulain et al. (2023)
            -- Multi-compartmental model of glymphatic clearance of solutes in brain tissue
            (https://doi.org/10.1371/journal.pone.0280501) to generate meshes of a ratbrain. It includes
            python-scripts for generating a FEniCS-compatible meshes from the included stl-files, as well
            as a 3DSlicer-compatible (https://www.slicer.org/) segmentation file that were used to generate
             the stls.""",
            doi="10.5281/zenodo.10076317",
            license="CC-BY-4.0",
            links={
                "brain.stl": "https://zenodo.org/records/8138343/files/brain.stl?download=1",
                "environment.yml": "https://zenodo.org/records/8138343/files/environment.yml?download=1",
                "LICENSE.txt": "https://zenodo.org/records/8138343/files/LICENSE.txt?download=1",
                "mesh_generation.py": "https://zenodo.org/records/8138343/files/mesh_generation.py?download=1",
                "meshprocessing.py": "https://zenodo.org/records/8138343/files/meshprocessing.py?download=1",
                "README.md": "https://zenodo.org/records/8138343/files/README.md?download=1",
                "segmentation.seg.nrrd": "https://zenodo.org/records/8138343/files/segmentation.seg.nrrd?download=1",
                "ventricles.stl": "https://zenodo.org/records/8138343/files/ventricles.stl?download=1",
            },
        ),
    }


# From https://gist.github.com/maxpoletaev/521c4ce2f5431a4afabf19383fc84fe2
class ProgressBar:
    def __init__(self, filename: str):
        self.tqdm = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if self.tqdm is None:
            self.tqdm = tqdm.tqdm(
                total=total_size,
                unit_divisor=1024,
                unit_scale=True,
                unit="B",
                desc=self.filename,
                leave=False,
            )

        progress = block_num * block_size
        if progress >= total_size:
            self.tqdm.close()
            return

        self.tqdm.update(progress - self.tqdm.n)


def list_datasets():
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()
    datasets = get_datasets()

    console.print(Panel.fit("[bold cyan]Available Datasets[/bold cyan]", border_style="cyan"))

    for key, dataset in datasets.items():
        # Create a table for the files/links
        link_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", expand=True)
        link_table.add_column("Filename", style="green")
        link_table.add_column("URL", style="blue", overflow="fold")

        for filename, url in dataset.links.items():
            link_table.add_row(filename, url)

        # Format description text (strip whitespace for cleaner output)
        description_text = Text(
            dataset.description.strip().replace("            ", "").replace("\n", " "),
            style="white",
        )

        # Create the main content grid
        content = Table.grid(padding=1)
        content.add_column(style="bold yellow", justify="right")
        content.add_column(style="white")

        content.add_row("Key:", key)
        content.add_row("DOI:", dataset.doi)
        content.add_row("License:", dataset.license)
        content.add_row("Description:", description_text)
        content.add_row("Files:", link_table)

        # Wrap in a Panel
        console.print(
            Panel(
                content,
                title=f"[bold]{dataset.name}[/bold]",
                subtitle=f"[dim]Key: {key}[/dim]",
                border_style="green",
                expand=True,
            )
        )
        console.print("")  # Add spacing between panels


def add_arguments(parser):
    subparsers = parser.add_subparsers(dest="datasets-command")
    download_parser = subparsers.add_parser("download", help="Download a dataset", formatter_class=parser.formatter_class)
    choices = list(get_datasets().keys())
    download_parser.add_argument(
        "dataset",
        type=str,
        default=choices[0],
        choices=choices,
        help=f"Dataset to download (choices: {', '.join(choices)})",
    )
    download_parser.add_argument("-o", "--outdir", type=Path, help="Output directory to download test data")

    subparsers.add_parser("list", help="List available datasets")


def dispatch(args):
    subcommand = args.pop("datasets-command", None)
    if subcommand == "list":
        list_datasets()
        return
    elif subcommand == "download":
        dataset = args.pop("dataset")
        outdir = args.pop("outdir")
        if outdir is None:
            logger.error("Output directory (-o or --outdir) is required for downloading datasets.")
            return

        datasets = get_datasets()
        if dataset not in datasets:
            logger.error(f"Unknown dataset: {dataset}. Available datasets: {', '.join(datasets.keys())}")
            return

        links = datasets[dataset].links
        download_multiple(links, outdir)
    else:
        raise ValueError(f"Unknown subcommand: {subcommand}")


# def download_data(outdir: Path, file_info: tuple) -> None:
def download_data(args) -> Path:
    (outdir, file_info) = args
    (filename, url) = file_info
    output_path = outdir / Path(filename).stem / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} to {output_path}. This may take a while.")

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=ProgressBar(filename=filename))
        if not zipfile.is_zipfile(output_path):
            logger.info(f"Downloaded {filename} is not a zip file. No extraction needed.")
            return output_path
        logger.info(f"Extracting {output_path} to {output_path.parent}.")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(output_path.parent)
        output_path.unlink()
    except Exception as e:
        logger.error(f"Failed to download {filename} from {url}. Error: {e}")
        raise
    return output_path


# Download multiple files concurrently
# Implementation inspired by https://medium.com/@ryan_forrester_/downloading-files-from-urls-in-python-f644e04a0b16
def download_multiple(urls: dict, outdir, max_workers=1):
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for thread pool
    args = [(outdir, file_info) for file_info in urls.items()]

    # Download files using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(download_data, args),
                total=len(urls),
                desc="Downloading MRI data",
            )
        )

    # Process results
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)

    print("\nDownload complete:")
    print(f"- Successfully downloaded: {len(successful)} files")
    print(f"- Failed downloads: {failed} files")

    return successful
