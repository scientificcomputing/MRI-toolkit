# MRI-Toolkit

`MRI-toolkit` provides a set of features dedicated to human MRI data post-processing and analysis. The implementation is based on [gMRI2FEM](https://github.com/jorgenriseth/gMRI2FEM).

## Installation

```bash
pip install mritk
```

## Documentation

The documentation is available at [https://scientificcomputing.github.io/mri-toolkit/](https://scientificcomputing.github.io/mri-toolkit/). It includes detailed usage instructions, API references, and examples.


## Quick Start

To get started with `mri-toolkit`, you can use the command-line interface (CLI) to inspect and analyze your MRI data.

![readme](https://github.com/user-attachments/assets/404bc4be-7267-4d1c-9126-0bee7c4a316c)


## Features

* **File Inspection**: detailed NIfTI header analysis (affine, voxel size, shape).
* **Statistics**: Compute comprehensive statistics (volume, mean, median, std, percentiles) for MRI regions based on segmentation maps.
* **Visualization**:
    * **Terminal**: View orthogonal slices (Sagittal, Coronal, Axial) directly in your console.
    * **Napari**: Launch the Napari viewer for interactive 3D inspection.
* **Data Management**: Utilities to download test datasets.
