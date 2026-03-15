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
# Concentration Mapping

The Concentration module calculates the spatial distribution of a contrast agent (like gadobutrol) in the brain.

Concentration $C$ can be estimated voxel-wise from longitudinal relaxation data comparing a post-contrast session to a pre-contrast (baseline) session. mritk supports two calculation pathways depending on whether you are working with $T_1$ times or $R_1$ rates (see R1 Maps).

$$\frac{1}{T_1} = \frac{1}{T_{10}} + r_1 C \quad \implies \quad C = \frac{1}{r_1} \left(R_1 - R_{10}\right)$$

where $r_1$ is the relaxivity of the contrast agent (default: 3.2 to 4.5 $\text{s}^{-1}\text{mmol}^{-1}$).

## Pipeline Overview

```{mermaid}
graph TD
    A[Pre-Contrast Hybrid T1] -->|T1 method| C{Compute Concentration}
    B[Post-Contrast Hybrid T1] -->|T1 method| C

    A2[Pre-Contrast R1] -->|R1 method| C
    B2[Post-Contrast R1] -->|R1 method| C

    M[Brain/Intracranial Mask] -.->|Optional| C

    C --> D(Tracer Concentration Map NIfTI)
```

## Commands

```{code-cell} shell
!mritk concentration --help
```


### 1. From $T_1$ Maps (t1)

Calculates concentration directly from $T_1$ maps (in milliseconds). The command handles the inversion safely and avoids division-by-zero errors for background voxels.

```{code-cell} shell
!mritk concentration t1 --help
```

#### Example Command

```shell
mritk concentration t1 -i path/to/post_t1.nii.gz -r path/to/pre_t1.nii.gz -o path/to/concentration.nii.gz --r1 0.0045 --mask path/to/intracranial_mask.nii.gz
```

Gonzo:

```shell
mritk concentration t1 \
    -i gonzo/mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.nii.gz \
    -r gonzo/mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-01_T1map_hybrid.nii.gz \
    -o sub-01_ses-02_concentration.nii.gz \
    --r1 0.0032 \
    --mask gonzo/mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-intracranial_binary.nii.gz
```



### 2. From $R_1$ Maps (r1)

Calculates concentration from pre-computed $R_1$ maps. This is mathematically equivalent but slightly faster if $R_1$ maps are already available.

```{code-cell} shell
!mritk concentration r1 --help
```

#### Example Command

```shell
mritk concentration r1 -i path/to/post_r1.nii.gz -r path/to/pre_r1.nii.gz -o path/to/concentration.nii.gz --r1 0.0045
```
