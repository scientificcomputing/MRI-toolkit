# MRI-Toolkit

`MRI-toolkit` provides a set of features dedicated to MRI data post-processing and analysis.

The implementation is inspired by [gMRI2FEM](https://github.com/jorgenriseth/gMRI2FEM), and some of the code is taken from that project. However, `MRI-toolkit` is designed to be more modular and extensible, with a focus on providing a user-friendly command-line interface (CLI) for common MRI processing tasks.

## Installation

```bash
pip install mritk
```

## Documentation

The documentation is available at [https://scientificcomputing.github.io/mri-toolkit/](https://scientificcomputing.github.io/mri-toolkit/). It includes detailed usage instructions, API references, and examples.


## Quick Start

To get started with `mri-toolkit`, you can use the command-line interface (CLI) to inspect and analyze your MRI data.

https://github.com/user-attachments/assets/5643e455-c7e6-4021-89bf-4fc60729be28


## Features


- File Inspection: detailed NIfTI header analysis (affine, voxel size, shape).

- $T_1$ Mapping: Estimate $T_1$ relaxation times using Look-Locker or Mixed sequences, and seamlessly merge them into comprehensive Hybrid $T_1$ maps.

- $R_1$ Relaxation Rates: Convert $T_1$ maps into $R_1$ relaxation rate maps for linear scaling with tracer concentrations.

- Concentration Mapping: Calculate the spatial distribution of contrast agents (e.g., gadobutrol) utilizing pre- and post-contrast $T_1$ or $R_1$ maps.

- Statistics: Compute comprehensive statistics (volume, mean, median, std, percentiles) for MRI regions based on segmentation maps.

- Visualization:

    - Terminal: View orthogonal slices (Sagittal, Coronal, Axial) directly in your console.

    - Napari: Launch the Napari viewer for interactive 3D inspection.

- Data Management: Utilities to download datasets.

## Contributing
Contributions to `MRI-toolkit` are welcome! If you have an idea for a new feature, improvement, or bug fix, please open an issue or submit a pull request on GitHub. For more details on how to contribute, please see the [Contributing Guide](CONTRIBUTING.md).

## License
`MRI-toolkit` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
