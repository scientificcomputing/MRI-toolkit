---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

 # Show Command

 The `show` command provides a quick way to visualize MRI data directly in your terminal.
 It displays three orthogonal slices: Sagittal, Coronal, and Axial.

 :::{note}

 This command requires the optional `[show]` dependencies, which include `textual-image`, `pillow` and `matplotlib`. You can install these with the following command:

 ```bash
 pip install mritk[show]
 ```

 :::

 ## Usage

 ```{code-cell} shell
 !mritk show --help
 ```

 ### Example Command

 ```bash
 mritk show path/to/image.nii.gz
 ```

 %% [markdown]
 ## Example

 View the center of the brain using the default gray colormap:

 ```bash
 mritk show data/mri.nii.gz
 ```

 View specific slices with a different colormap (e.g., `viridis` or `magma` if matplotlib is installed):

 ```bash
 mritk show data/mri.nii.gz --cmap viridis --slice-z 0.3
 ```

 ![show](https://github.com/user-attachments/assets/cd037567-3df8-4ad9-9f94-70478edf6e5e)
