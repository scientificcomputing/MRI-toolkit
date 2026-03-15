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

# Hybrid $T_1$ Maps

To achieve accurate $T_1$ measurements across the entire brain space, mritk combines the Look-Locker (LL) and Mixed sequence $T_1$ maps into a single Hybrid map.

Look-Locker is used for short $T_1$ values (brain tissue and regions with high tracer concentrations).

Mixed Sequence is used for long $T_1$ values (CSF).

The hybrid command seamlessly merges the two images based on a user-defined threshold (default: 1500 ms) and a specific anatomical mask (typically a CSF mask).

## Pipeline Overview

```{mermaid}
graph LR
    A(Look-Locker T1 Map) --> D{Hybrid Merge}
    B(Mixed T1 Map) --> D
    C(CSF Mask) --> D
    D -->|Threshold > 1500ms| E(Hybrid T1 Map)
```

## Command Usage


```{code-cell} shell
!mritk hybrid --help
```

## Example Command

```shell
mritk hybrid -l path/to/ll_t1.nii.gz -m path/to/mixed_t1.nii.gz -c path/to/csf_mask.nii.gz -o path/to/hybrid_t1.nii.gz --threshold 1500.0
```


Gonzo:

```shell
mritk hybrid \
    -l gonzo/mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-02_acq-looklocker_T1map_registered.nii.gz \
    -m gonzo/mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-02_acq-mixed_T1map_registered.nii.gz \
    -c gonzo/mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf_binary.nii.gz \
    -o sub-01_ses-02_T1map_hybrid.nii.gz \
    --threshold 1500 \
    --erode 1
```
