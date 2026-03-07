# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import json
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import scipy
import scipy.interpolate
import skimage
import nibabel


from ..data.orientation import data_reorientation, change_of_coordinates_map

from ..data.base import MRIData
from ..data.io import load_mri_data
from ..masking.masks import create_csf_mask
from .utils import T1_lookup_table, VOLUME_LABELS, run_dcm2niix

logger = logging.getLogger(__name__)


def dicom_standard_affine(frame_fg) -> np.ndarray:
    """
    Generates the DICOM to LPS (Left-Posterior-Superior) affine transformation matrix.

    This maps the voxel coordinate space of a DICOM frame to the physical LPS space
    by utilizing the pixel spacing, slice spacing, and patient orientation cosines.

    Args:
        frame_fg: A DICOM frame functional group sequence object containing
            PixelMeasuresSequence, PlaneOrientationSequence, and PlanePositionSequence.

    Returns:
        np.ndarray: A 4x4 affine transformation matrix mapping from DICOM voxel
        indices to LPS physical coordinates.
    """
    # Get the original data shape
    df = float(frame_fg.PixelMeasuresSequence[0].SpacingBetweenSlices)
    dr, dc = (float(x) for x in frame_fg.PixelMeasuresSequence[0].PixelSpacing)
    plane_orientation = frame_fg.PlaneOrientationSequence[0]
    orientation = np.array(plane_orientation.ImageOrientationPatient)

    # Find orientation of data array relative to LPS-coordinate system.
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    frame_cosine = np.cross(row_cosine, col_cosine)

    # Create DICOM-definition affine map to LPS.
    T_1 = np.array(frame_fg.PlanePositionSequence[0].ImagePositionPatient)

    # Create DICOM-definition affine map to LPS.
    M_dcm = np.zeros((4, 4))
    M_dcm[:3, 0] = row_cosine * dc
    M_dcm[:3, 1] = col_cosine * dr
    M_dcm[:3, 2] = frame_cosine * df
    M_dcm[:3, 3] = T_1
    M_dcm[3, 3] = 1.0

    # Reorder from "natural index order" to DICOM affine map definition order.
    N_order = np.eye(4)[[2, 1, 0, 3]]
    return M_dcm @ N_order


def extract_single_volume(D: np.ndarray, frame_fg) -> MRIData:
    """
    Extracts, scales, and reorients a single DICOM volume into an MRIData object.

    Applies the appropriate RescaleSlope and RescaleIntercept transformations
    to the raw pixel array, and then reorients the resulting data volume from
    the native DICOM LPS space to RAS (Right-Anterior-Superior) space.

    Args:
        D (np.ndarray): The raw 3D pixel array for the volume.
        frame_fg: The corresponding DICOM frame functional group metadata.

    Returns:
        MRIData: A newly constructed MRIData object with scaled pixel values
        and an affine matrix oriented to RAS space.
    """
    # Find scaling values (should potentially be inside scaling loop)
    pixel_value_transform = frame_fg.PixelValueTransformationSequence[0]
    slope = float(pixel_value_transform.RescaleSlope)
    intercept = float(pixel_value_transform.RescaleIntercept)
    private = frame_fg[0x2005, 0x140F][0]
    scale_slope = private[0x2005, 0x100E].value

    # Loop over and scale values.
    volume = np.zeros_like(D, dtype=np.single)
    for idx in range(D.shape[0]):
        volume[idx] = (intercept + slope * D[idx]) / (scale_slope * slope)

    A_dcm = dicom_standard_affine(frame_fg)
    C = change_of_coordinates_map("LPS", "RAS")
    mri = data_reorientation(MRIData(volume, C @ A_dcm))

    return mri


def mixed_t1map(
    SE_nii_path: Path, IR_nii_path: Path, meta_path: Path, T1_low: float, T1_high: float, output: Path | None = None
) -> nibabel.nifti1.Nifti1Image:
    """
    Generates a T1 relaxation map by combining Spin-Echo (SE) and Inversion-Recovery (IR) acquisitions.

    This function acts as an I/O wrapper. It loads the respective NIfTI volumes
    and their sequence metadata (such as TR, TE, TI, and Echo Train Length),
    and passes them to the underlying mathematical function which interpolates
    the T1 values based on the theoretical signal ratio (IR/SE).

    Args:
        SE_nii_path (Path): Path to the Spin-Echo modulus NIfTI file.
        IR_nii_path (Path): Path to the Inversion-Recovery corrected real NIfTI file.
        meta_path (Path): Path to the JSON file containing the sequence parameters
            ('TR_SE', 'TI', 'TE', 'ETL').
        T1_low (float): Lower bound for the T1 interpolation grid (in ms).
        T1_high (float): Upper bound for the T1 interpolation grid (in ms).
        output (Path | None, optional): Path to save the resulting T1 map NIfTI file. Defaults to None.

    Returns:
        nibabel.nifti1.Nifti1Image: The computed T1 map as a NIfTI image object,
        with the qform/sform properly set to scanner space.
    """
    se_mri = load_mri_data(SE_nii_path, dtype=np.single)
    ir_mri = load_mri_data(IR_nii_path, dtype=np.single)
    meta = json.loads(meta_path.read_text())

    t1_volume = compute_mixed_t1_array(se_mri.data, ir_mri.data, meta, T1_low, T1_high)

    nii = nibabel.nifti1.Nifti1Image(t1_volume, ir_mri.affine)
    nii.set_sform(nii.affine, "scanner")
    nii.set_qform(nii.affine, "scanner")

    if output is not None:
        nibabel.nifti1.save(nii, output)

    return nii


def mixed_t1map_postprocessing(SE_nii_path: Path, T1_path: Path, output: Path | None = None) -> nibabel.nifti1.Nifti1Image:
    """
    Masks a Mixed T1 map to isolate the Cerebrospinal Fluid (CSF).

    Because the Mixed sequence is primarily sensitive/calibrated for long T1 species
    like fluid, this function isolates the CSF. It derives a mask dynamically from
    the original Spin-Echo sequence using Li thresholding, erodes the mask to avoid
    partial-voluming effects at tissue boundaries, and applies it to the T1 map.

    Args:
        SE_nii_path (Path): Path to the Spin-Echo NIfTI file used to derive the mask.
        T1_path (Path): Path to the previously generated Mixed T1 map NIfTI file.
        output (Path | None, optional): Path to save the masked T1 NIfTI file. Defaults to None.

    Returns:
        nibabel.nifti1.Nifti1Image: The masked T1 map, where all non-CSF voxels
        have been set to NaN.
    """
    t1map_nii = nibabel.nifti1.load(T1_path)
    se_mri = load_mri_data(SE_nii_path, np.single)

    mask = create_csf_mask(se_mri.data, use_li=True)
    mask = skimage.morphology.erosion(mask)

    masked_t1map = t1map_nii.get_fdata(dtype=np.single)
    masked_t1map[~mask] = np.nan
    masked_t1map_nii = nibabel.nifti1.Nifti1Image(masked_t1map, t1map_nii.affine, t1map_nii.header)

    if output is not None:
        nibabel.nifti1.save(masked_t1map_nii, output)

    return masked_t1map_nii


def compute_mixed_t1_array(se_data: np.ndarray, ir_data: np.ndarray, meta: dict, t1_low: float, t1_high: float) -> np.ndarray:
    """
    Computes a Mixed T1 array from Spin-Echo and Inversion-Recovery volumes using a lookup table.

    Args:
        se_data (np.ndarray): 3D numpy array of the Spin-Echo modulus data.
        ir_data (np.ndarray): 3D numpy array of the Inversion-Recovery corrected real data.
        meta (dict): Dictionary containing sequence parameters ('TR_SE', 'TI', 'TE', 'ETL').
        t1_low (float): Lower bound for T1 generation grid.
        t1_high (float): Upper bound for T1 generation grid.

    Returns:
        np.ndarray: Computed T1 map as a 3D float32 array.
    """
    nonzero_mask = se_data != 0
    f_data = np.nan * np.zeros_like(ir_data)
    f_data[nonzero_mask] = ir_data[nonzero_mask] / se_data[nonzero_mask]

    tr_se, ti, te, etl = meta["TR_SE"], meta["TI"], meta["TE"], meta["ETL"]
    f_curve, t1_grid = T1_lookup_table(tr_se, ti, te, etl, t1_low, t1_high)

    interpolator = scipy.interpolate.interp1d(f_curve, t1_grid, kind="nearest", bounds_error=False, fill_value=np.nan)
    return interpolator(f_data).astype(np.single)


def _extract_frame_metadata(frame_fg) -> dict:
    """
    Extracts core physical parameters (TR, TE, TI, ETL) from a DICOM frame functional group.

    Args:
        frame_fg: The PerFrameFunctionalGroupsSequence element for a specific frame.

    Returns:
        dict: A dictionary containing available MR timing parameters.
    """
    descrip = {
        "TR": float(frame_fg.MRTimingAndRelatedParametersSequence[0].RepetitionTime),
        "TE": float(frame_fg.MREchoSequence[0].EffectiveEchoTime),
    }

    if hasattr(frame_fg.MRModifierSequence[0], "InversionTimes"):
        descrip["TI"] = frame_fg.MRModifierSequence[0].InversionTimes[0]

    if hasattr(frame_fg.MRTimingAndRelatedParametersSequence[0], "EchoTrainLength"):
        descrip["ETL"] = frame_fg.MRTimingAndRelatedParametersSequence[0].EchoTrainLength

    return descrip


def extract_mixed_dicom(dcmpath: Path, subvolumes: list[str]) -> list[dict]:
    """
    Reads a Mixed DICOM file and splits it into independent NIfTI subvolumes.

    Args:
        dcmpath (Path): Path to the input DICOM file.
        subvolumes (list[str]): List of volume labels mapping to the slices in the DICOM.

    Returns:
        list[dict]: A list containing dictionaries with a generated 'nifti' image
        and a 'descrip' metadata dictionary for each requested subvolume.
    """
    import pydicom

    dcm = pydicom.dcmread(str(dcmpath))
    frames_total = int(dcm.NumberOfFrames)

    # [0x2001, 0x1018] is a private Philips tag representing 'Number of Slices MR'
    frames_per_volume = dcm[0x2001, 0x1018].value
    num_volumes = frames_total // frames_per_volume
    assert num_volumes * frames_per_volume == frames_total, "Subvolume dimensions do not evenly divide the total frames."

    pixel_data = dcm.pixel_array.astype(np.single)
    frame_fg_sequence = dcm.PerFrameFunctionalGroupsSequence

    vols_out = []
    for volname in subvolumes:
        vol_idx = VOLUME_LABELS.index(volname)

        # Find volume slices representing the current subvolume
        subvol_idx_start = vol_idx * frames_per_volume
        subvol_idx_end = (vol_idx + 1) * frames_per_volume
        frame_fg = frame_fg_sequence[subvol_idx_start]

        logger.info(
            f"Converting volume {vol_idx + 1}/{len(VOLUME_LABELS)}: '{volname}' "
            f"between indices {subvol_idx_start}-{subvol_idx_end} out of {frames_total}."
        )

        mri = extract_single_volume(pixel_data[subvol_idx_start:subvol_idx_end], frame_fg)

        nii_oriented = nibabel.nifti1.Nifti1Image(mri.data, mri.affine)
        nii_oriented.set_sform(nii_oriented.affine, "scanner")
        nii_oriented.set_qform(nii_oriented.affine, "scanner")

        description = _extract_frame_metadata(frame_fg)
        vols_out.append({"nifti": nii_oriented, "descrip": description})

    return vols_out


def dicom_to_mixed(dcmpath: Path, outpath: Path, subvolumes: Optional[list[str]] = None):
    """
    Converts a Mixed sequence DICOM file into independent subvolume NIfTIs.

    Generates dedicated images for Spin-Echo, Inversion-Recovery, etc.,
    and saves sequence timing metadata to a JSON sidecar.

    Args:
        dcmpath (Path): Path to the input Mixed DICOM file.
        outpath (Path): Base path for output files. Suffixes are automatically appended.
        subvolumes (list[str], optional): specific subvolumes to extract.
            Defaults to all known VOLUME_LABELS.
    """
    subvolumes = subvolumes or VOLUME_LABELS
    assert all([volname in VOLUME_LABELS for volname in subvolumes]), (
        f"Invalid subvolume name in {subvolumes}, must be one of {VOLUME_LABELS}"
    )

    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    vols = extract_mixed_dicom(dcmpath, subvolumes)
    meta = {}

    for vol, volname in zip(vols, subvolumes):
        output = outpath.with_name(f"{outpath.stem}_{volname}.nii.gz")
        nibabel.nifti1.save(vol["nifti"], output)

        descrip = vol["descrip"]
        try:
            if volname == "SE-modulus":
                meta["TR_SE"] = descrip["TR"]
                meta["TE"] = descrip["TE"]
                meta["ETL"] = descrip["ETL"]
            elif volname == "IR-corrected-real":
                meta["TR_IR"] = descrip["TR"]
                meta["TI"] = descrip["TI"]
        except KeyError as e:
            logger.error(f"Missing required metadata for {volname}: {descrip}")
            raise e

    # Write merged metadata sidecar
    (outdir / f"{form}_meta.json").write_text(json.dumps(meta, indent=4))

    # Attempt standard dcm2niix conversion (soft failure allowed for legacy behavior)
    run_dcm2niix(dcmpath, outdir, form, extra_args="-w 0 --terse -b o", check=False)
