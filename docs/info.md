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

# Info command

The `info` command allows you to quickly inspect the metadata of an MRI file. It displays the image shape, voxel size, data type, and the affine transformation matrix.

## Usage

```{code-cell} shell
!mritk info --help
```

### Example Command

```bash
mritk info path/to/image.nii.gz
```


![info](https://github.com/user-attachments/assets/fc0e734d-3c94-48fa-8e25-3e65bfc86ebe)
