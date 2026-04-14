# Selecting a region of interest (ROI)

```python
from pathlib import Path
import logging
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import nibabel as nib
import mritk
```

First we make sure that the necessary data is downloaded and present in a folder called "gonzo" in the same directory as this script

```python
mri_data_dir = Path("gonzo")
```

If you haven't downloaded he gonzo dataset, you can do so with the following command:
mritk datasets download gonzo
```shell
 mritk datasets download gonzo -o gonzo
```

If you want to learn more about the gonzo dataset, you can check the command
```shell
mritk datasets info gonzo
```
which will point you do the url https://doi.org/10.5281/zenodo.14266867.


```python
# We now load the full T1-weighted image
t1_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
t1_data = mritk.MRIData.from_file(t1_path)
```

We define a new grid of points in physical space that we want to extract from the original image.
This grid is defined by the ranges of x, y, and z coordinates and the number of points along each axis.

```python
xs = np.linspace(0, 70, 80)
ys = np.linspace(0, 20, 20)
zs = np.linspace(-40, 90, 110)
```

```python
# Create a 3D grid of points as one long vector
grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
```

```python
# We also define a new affine transformation for the extracted piece, which in this case is just the identity matrix. This means that the extracted piece will be in the same physical space as the original image.
new_affine = np.eye(4)
new_affine[0, 0] = xs[1] - xs[0]
new_affine[1, 1] = ys[1] - ys[0]
new_affine[2, 2] = zs[1] - zs[0]
new_affine[0, 3] = xs[0]
new_affine[1, 3] = ys[0]
new_affine[2, 3] = zs[0]
```

We now compute the corresponding voxel indices in the original image for each point in our grid using the affine transformation of the original image.

```python
vi = mritk.data.physical_to_voxel_indices(grid_points, affine=t1_data.affine)  # Shape: (N, 3)
```

Finally, we extract the values at the specified voxel indices and reshape them back to the original grid shape.

```python
v = t1_data.data[tuple(vi.T)]
v = v.reshape(grid_x.shape)
```

```python
# We save the extracted values as a new NIfTI file with the same affine as the original image.
piece_t1_data = mritk.MRIData(data=v, affine=new_affine)
piece_t1_data.save("piece_T1.nii.gz")
```

We can now visualize the extracted piece of the T1 image using napari or any other NIfTI viewer. The extracted piece will correspond to the specified grid of points in physical space, allowing us to focus on a specific region of interest (ROI) within the original image.
We can do this using the command:
```shell
mritk napari piece_T1.nii.gz gonzo/mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz
```
![piece_t1](https://github.com/user-attachments/assets/242db608-e9f8-4e4d-8d9e-4c6acc5da94f)

We can now try to do the same thing for the concentration files, which are stored in a different folder and have a different naming convention. We will loop through all the concentration files, extract the values at the specified voxel indices,
and save them as new NIfTI files with the same affine as the original image.

```python
concentration_files = list(
        sorted(
            (mri_data_dir / "mri-processed/mri_processed_data/sub-01/concentrations").glob("sub-01_ses-0*_concentration.nii.gz")
        )
    )
```

```python
vs = []
```

```python
for i, path in enumerate(concentration_files):
    print(path)
    conc_data = mritk.MRIData.from_file(path)
    vi = mritk.data.physical_to_voxel_indices(grid_points, affine=conc_data.affine)  # Shape: (N, 3)
    v = conc_data.data[tuple(vi.T)]  # Extract values at the specified voxel indices
    v = v.reshape(grid_x.shape)
    vs.append(v)
```

```python
piece_data = mritk.MRIData(data=np.stack(vs), affine=new_affine)
piece_data.save(f"piece_conc.nii.gz")
```

We can visualize the extracted concentration files in napari as well, using the command:
```shell
mritk napari piece_conc.nii.gz piece_T1.nii.gz "gonzo/mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
```
![conc_t1](https://github.com/user-attachments/assets/f306e30a-8b12-480c-9bb4-94a4089696a0)
