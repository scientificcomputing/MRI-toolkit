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

# Mixed Sequence $T_1$ Mapping

The Mixed sequence module estimates $T_1$ times by combining a Spin-Echo (SE) and an Inversion-Recovery (IR) acquisition.

While the Look-Locker sequence struggles with long relaxation times due to short acquisition windows, the Mixed sequence is specifically designed to accurately estimate long $T_1$ times, such as those found in Cerebrospinal Fluid (CSF). $T_1$ is estimated by solving the non-linear ratio of the IR and SE signals.

```{caution}
Because the Mixed sequence is highly sensitive to noise in short $T_1$ tissues (like gray matter), the resulting $T_1$ map must be post-processed to mask out non-CSF areas.
```

## Pipeline Overview

```{mermaid}
graph TD
    A[Raw Mixed DICOM] -->|dcm2mixed| B(SE Modulus NIfTI)
    A -->|dcm2mixed| C(IR Real NIfTI)
    A -->|dcm2mixed| D(Metadata JSON)
    B -->|t1| E(Raw Mixed T1 Map)
    C -->|t1| E
    D -->|t1| E
    E -->|postprocess| F(Masked CSF T1 Map)
    B -->|postprocess| F
```

## Commands


```{code-cell} shell
!mritk mixed --help
```

### 1. DICOM to NIfTI (dcm2mixed)

Splits a Mixed sequence DICOM file into its independent subvolumes (e.g., SE-modulus, IR-real) and saves the required sequence timing metadata (TR, TE, TI, ETL) into a JSON sidecar.

```{code-cell} shell
!mritk mixed dcm2mixed --help
```

#### Example Command


```bash
mritk mixed dcm2mixed -i path/to/mixed.dcm -o path/to/output_base
```

### 2. Compute $T_1$ Map (t1)

Generates the $T_1$ map based on the signal ratio between the Inversion-Recovery and Spin-Echo sequences.

```{code-cell} shell
!mritk mixed t1 --help
```

#### Example Command

```bash
mritk mixed t1 -s path/to/output_base_SE-modulus.nii.gz -i path/to/output_base_IR-corrected-real.nii.gz -m path/to/output_base_meta.json -o path/to/mixed_t1_raw.nii.gz
```

Gonzo:

```shell
mritk mixed t1 \
    -s gonzo/mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_SE-modulus.nii.gz \
    -i gonzo/mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_IR-corrected-real.nii.gz \
    -m gonzo/mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_meta.json \
    -o sub-01_ses-01_acq-mixed_T1map_raw.nii.gz
```

### 3. Post-Processing (postprocess)

Masks out non-fluid areas from the Mixed $T_1$ map. It derives a mask dynamically from the original SE sequence using Li thresholding and erodes the mask to avoid partial-volume effects at tissue boundaries.

```{code-cell} shell
!mritk mixed postprocess --help
```

#### Example Command
```bash
mritk mixed postprocess -s path/to/output_base_SE-modulus.nii.gz -t path/to/mixed_t1_raw.nii.gz -o path/to/mixed_t1_clean.nii.gz
```

Gonzo:

(here we use the original SE modulus image as the source for mask generation, and the t1 map from the previous step as the input for post-processing)

```shell
mritk mixed postprocess \
    -s gonzo/mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_SE-modulus.nii.gz \
    -t sub-01_ses-01_acq-mixed_T1map_raw.nii.gz \
    -o sub-01_ses-01_acq-mixed_T1map_clean.nii.gz
```
