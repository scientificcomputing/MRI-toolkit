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

# Look-Locker $T_1$ Mapping

The Look-Locker (LL) module is used to estimate $T_1$ relaxation times from a 4D Look-Locker inversion recovery dataset.

As described in the Gonzo dataset, the Look-Locker sequence provides excellent accuracy for tissues with short $T_1$ times (such as gray/white matter and regions with high tracer concentrations). The toolkit computes the $T_1$ time voxel-wise by fitting a theoretical recovery curve to the longitudinal magnetization signal.

## Pipeline Overview

```{mermaid}
graph TD
    A[Raw Look-Locker DICOM] -->|dcm2ll| B(4D Look-Locker NIfTI)
    A -->|dcm2ll| C(Trigger Times .txt)
    B -->|t1| D(Raw T1 Map NIfTI)
    C -->|t1| D
    D -->|postprocess| E(Cleaned T1 Map NIfTI)
```

## Commands

```{code-cell} shell
!mritk looklocker --help
```


### 1. DICOM to NIfTI (dcm2ll)

Converts scanner-native Look-Locker DICOM files to a standardized 4D NIfTI format and extracts the nominal cardiac trigger delay times into a sidecar text file.

```{code-cell} shell
!mritk looklocker dcm2ll --help
```

#### Example Command

```bash
mritk looklocker dcm2ll -i path/to/looklocker.dcm -o path/to/ll_output.nii.gz
```

### 2. Compute $T_1$ Map (t1)

Fits the voxel-wise Levenberg-Marquardt optimization curve to estimate $T_1$ times (in milliseconds) from the 4D Look-Locker NIfTI.

```{code-cell} shell
!mritk looklocker t1 --help
```

#### Example Command

```bash
mritk looklocker t1 -i path/to/ll_output.nii.gz -t path/to/ll_output_trigger_times.txt -o path/to/t1_raw.nii.gz
```

Gonzo:

```shell
mritk looklocker t1 \
    -i gonzo/mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1.nii.gz \
    -t gonzo/mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1_trigger_times.txt \
    -o sub-01_ses-01_acq-looklocker_T1map_raw.nii.gz
```

### 3. Post-Processing (postprocess)

Raw $T_1$ maps often contain noisy fits or values outside physiological boundaries. The postprocess command applies a quality-control pipeline that:

Automatically masks the brain/head (if no explicit mask is provided).

Removes extreme outliers (default bounds: 100 ms to 10000 ms).

Iteratively fills internal NaNs (holes) using a smart Gaussian filter.

```{code-cell} shell
!mritk looklocker postprocess --help
```

#### Example Command

```bash
mritk looklocker postprocess -i path/to/t1_raw.nii.gz -o path/to/t1_clean.nii.gz --t1-low 100.0 --t1-high 5000.0
```

Gonzo:

(here input is the raw T1 map from the previous step)

```shell
mritk looklocker postprocess \
    -i sub-01_ses-01_acq-looklocker_T1map_raw.nii.gz \
    -o sub-01_ses-01_acq-looklocker_T1map.nii.gz \
    --t1-low 100.0 \
    --t1-high 6000.0
```
