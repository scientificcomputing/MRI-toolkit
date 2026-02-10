# Installation

## Basic Installation

You can install the package directly from pypi using pip:

```bash
pip install mritk
```

Note that it is recommended to use a virtual environment when installing the package, to avoid conflicts with other packages. You can create a virtual environment using `venv`:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install mritk
```

## Installing extra dependencies

Some features of `mritk` require additional dependencies. You can install these extra dependencies using pip with the appropriate extras syntax.

* **The show command**: To use the `show` command for visualizing MRI data in the terminal, you need to install the `textual-image`, `pillow` and `matplotlib` packages. You can do this with the following command:

```bash
pip install mritk[show]
```

* **The napari command**: To use the `napari` command for visualizing MRI data in the [Napari viewer](https://napari.org/), you need to install the `napari` package. You can do this with the following command:

```bash
pip install mritk[napari]
```

## Development Installation

If you want to contribute to the development of `mritk`, you can install the package in editable mode. This allows you to make changes to the source code and have them reflected immediately without needing to reinstall the package (see also the Contributing guide for more info).

```bash
git clone https://github.com/scientificcomputing/mri-toolkit.git
cd mri-toolkit
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -e .
```

It is also possible the package directly from GitHub without cloning the repository:

```bash
pip install git+https://github.com/scientificcomputing/mri-toolkit.git
```

## Installation with pipx
If you intend to use the command line interface, you may want to install the package in an isolated environment using [`pipx`](https://pipx.pypa.io/latest/). You can do this with the following command:

```bash
pipx install mritk
```
