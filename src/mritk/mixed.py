# Mixed sequence

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import json
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import nibabel
import numpy as np
import scipy
import scipy.interpolate
import skimage

from .data import MRIData, change_of_coordinates_map, data_reorientation
from .masks import compute_csf_mask_array
from .utils import VOLUME_LABELS, run_dcm2niix

logger = logging.getLogger(__name__)


T = typing.TypeVar("T", np.ndarray, float)


class MetaDict(typing.TypedDict):
    TR_SE: float
    TI: float
    TE: float
    ETL: int


@dataclass
class Mixed:
    """Class representing the combined Spin-Echo and Inversion-Recovery
    acquisitions for Mixed T1 mapping.

    Args:
        SE: An MRIData object containing the Spin-Echo modulus data.
        IR: An MRIData object containing the Inversion-Recovery corrected real data.
        meta: A dictionary containing the sequence parameters ('TR_SE', 'TI', 'TE', 'ETL').
    """

    SE: MRIData
    IR: MRIData
    meta: MetaDict

    @classmethod
    def from_file(cls, se_path: Path, ir_path: Path, meta_path: Path) -> "Mixed":
        se_mri = MRIData.from_file(se_path, dtype=np.single)
        ir_mri = MRIData.from_file(ir_path, dtype=np.single)
        meta = json.loads(meta_path.read_text())
        return cls(SE=se_mri, IR=ir_mri, meta=meta)

    def t1_map(self, T1_low: float = 500.0, T1_high: float = 5000.0) -> "MixedT1":
        """Computes the T1 map from the Spin-Echo and Inversion-Recovery data using the provided sequence parameters.

        This function first computes the theoretical T1 lookup table based on the sequence parameters,
        then applies the T1 estimation to each voxel by interpolating the observed signal ratios onto the T1 grid.
        Finally, it constructs a NIfTI image of the T1 map with the appropriate affine transformation.

        Args:
            T1_low: The lower bound for the T1 interpolation grid (in ms). Defaults to 500 ms.
            T1_high: The upper bound for the T1 interpolation grid (in ms). Defaults to 5000 ms.

        Returns:
            MixedT1: An object containing the Spin-Echo data and the computed T1 map.
        """

        t1_volume = compute_mixed_t1_array(self.SE.data, self.IR.data, self.meta, T1_low, T1_high)
        logger.debug(
            f"Computed T1 volume with shape {t1_volume.shape} and T1 range ({np.nanmin(t1_volume)}, {np.nanmax(t1_volume)}) ms."
        )
        nii = nibabel.nifti1.Nifti1Image(t1_volume, self.IR.affine)
        nii.set_sform(nii.affine, "scanner")
        nii.set_qform(nii.affine, "scanner")
        return MixedT1(SE=self.SE, T1=nii)


@dataclass
class MixedT1:
    """Class representing the computed T1 map from the Mixed sequence, along with the original Spin-Echo data.

    Args:
         SE: An MRIData object containing the original Spin-Echo modulus data.
         T1: A NIfTI image containing the computed T1 map.
    """

    SE: MRIData
    T1: nibabel.nifti1.Nifti1Image | nibabel.nifti1.Nifti1Pair

    @classmethod
    def from_file(cls, se_path: Path, t1_path: Path) -> "MixedT1":
        se_mri = MRIData.from_file(se_path, dtype=np.single)
        t1_nii = nibabel.nifti1.load(t1_path)
        return cls(SE=se_mri, T1=t1_nii)

    def save(self, outpath: Path) -> None:
        """Saves the T1 map to a NIfTI file at the specified path.

        Args:
            outpath: The path where the T1 map NIfTI file should be saved.
        """
        nibabel.nifti1.save(self.T1, outpath)

    def postprocess(self) -> "MixedT1":
        """Applies post-processing to the Mixed T1 map to isolate the CSF.

        Because the Mixed sequence is primarily sensitive/calibrated for long T1 species
        like fluid, this function isolates the CSF. It derives a mask dynamically from
        the original Spin-Echo sequence using Li thresholding, erodes the mask to avoid
        partial-voluming effects at tissue boundaries, and applies it to the T1 map.

        Returns:
            nibabel.nifti1.Nifti1Image: The masked T1 map, where all non-CSF voxels
            have been set to NaN.
        """
        logger.debug("Creating CSF mask from SE image using Li thresholding and morphological erosion.")
        mask = compute_csf_mask_array(self.SE.data, use_li=True)
        logger.debug("Performing morphological erosion on the CSF mask to reduce partial volume effects.")
        mask = skimage.morphology.erosion(mask)

        logger.debug(f"Generated CSF mask with shape {mask.shape} and {mask.sum()} valid voxels.")
        masked_t1map = self.T1.get_fdata(dtype=np.single)
        masked_t1map[~mask] = np.nan
        masked_t1map_nii = nibabel.nifti1.Nifti1Image(masked_t1map, self.T1.affine, self.T1.header)

        return MixedT1(SE=self.SE, T1=masked_t1map_nii)


def estimate_se_free_relaxation_time(TRse: float, TE: float, ETL: int) -> float:
    """
    Computes the estimated free relaxation time following a Spin Echo image.

    Corrects the standard Repetition Time (TR) by accounting for the Effective
    Echo Time (TE), the Echo Train Length (ETL), and an adjustment for 20
    dummy preparation echoes.

    Args:
        TRse (float): Repetition time of the spin echo sequence (in ms).
        TE (float): Effective echo time (in ms).
        ETL (int): Echo train length.

    Returns:
        float: The corrected free relaxation time `TRfree`.
    """
    return TRse - TE * (1 + 0.5 * (ETL - 1) / (0.5 * (ETL + 1) + 20))


def T1_lookup_table(TRse: float, TI: float, TE: float, ETL: int, T1_low: float, T1_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a Fraction/T1 lookup table for mixed T1 mapping interpolations.

    Calculates the theoretical ratio of the Inversion Recovery signal (Sir) to
    the Spin Echo signal (Sse) over a highly discretized grid of physiological
    T1 relaxation times.

    Args:
        TRse (float): Spin-echo repetition time (in ms).
        TI (float): Inversion time (in ms).
        TE (float): Effective echo time (in ms).
        ETL (int): Echo train length.
        T1_low (float): Lower bound of the T1 grid (in ms).
        T1_hi (float): Upper bound of the T1 grid (in ms).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - fractionCurve (np.ndarray): The theoretical Sir/Sse signal ratios.
            - T1_grid (np.ndarray): The corresponding T1 values (in ms).
    """
    TRfree = estimate_se_free_relaxation_time(TRse, TE, ETL)
    T1_grid = np.arange(int(T1_low), int(T1_hi + 1))
    S_SE, S_IR = T1_to_mixed_signals(T1_grid, TR=TRfree, TI=TI)
    fractionCurve = S_IR / S_SE
    return fractionCurve, T1_grid


def T1_to_mixed_signals(T1: T, TR: float, TI: float) -> tuple[np.ndarray, np.ndarray]:
    """Computes the theoretical Spin-Echo and Inversion-Recovery signals for a given T1.

    Evaluates the standard signal equations for Spin-Echo and Inversion-Recovery
    sequences based on the provided T1 relaxation time, repetition time (TR),
    and inversion time (TI).

    Args:
        T1: The T1 relaxation time (in ms) for which to compute the signals.
        TR: The repetition time of the Spin-Echo sequence (in ms).
        TI: The inversion time of the Inversion-Recovery sequence (in ms).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - S_SE (np.ndarray): The theoretical Spin-Echo signal.
            - S_IR (np.ndarray): The theoretical Inversion-Recovery signal.

    Notes:
        The Spin-Echo signal is computed as:

        ..math::
            S_{SE} = 1 - e^{-TR / T1}

        The Inversion-Recovery signal is computed as:

        ..math::
            S_{IR} = 1 - (1 + S_{SE}) e^{-TI / T1}

    """
    S_SE = 1.0 * (1.0 - np.exp(-TR / T1))
    S_IR = 1.0 - (1.0 + S_SE) * np.exp(-TI / T1)
    return S_SE, S_IR


def T1_to_noisy_T1_mixed(
    T1_true: np.ndarray, TR: float, TI: float, f_grid: np.ndarray, t_grid: np.ndarray, sigma: float = 0.04
) -> np.ndarray:
    """Simulates noisy Mixed T1 estimation from true T1 values using lookup table interpolation.

    Args:
        T1_true: The true T1 relaxation times (in ms) for which to simulate the noisy estimates.
        TR: The repetition time of the Spin-Echo sequence (in ms).
        TI: The inversion time of the Inversion-Recovery sequence (in ms).
        f_grid: The precomputed grid of theoretical Sir/Sse ratios corresponding to the T1 values in t_grid.
        t_grid: The precomputed grid of T1 values (in ms) corresponding to the ratios in f_grid.
        sigma: The standard deviation of the Gaussian noise to be added to the signals. Defaults to 0.04.

    Returns:
        np.ndarray: The simulated noisy T1 estimates (in ms) obtained
        by interpolating the noisy signal ratios onto the T1 grid.

    Notes:
        This function first computes the theoretical Spin-Echo
        and Inversion-Recovery signals for the given true T1 values,
        adds Gaussian noise to the Inversion-Recovery signals and Rician
        noise to the Spin-Echo signals to simulate measurement variability,
        and then uses the precomputed lookup table (f_grid and t_grid) to
        interpolate the noisy signal ratios back to T1 estimates.
    """
    S_SE_t, S_IR_t = T1_to_mixed_signals(T1_true, TR, TI)
    real_SE = S_SE_t + np.random.normal(0, sigma, S_SE_t.shape)
    imag_SE = np.random.normal(0, sigma, S_SE_t.shape)
    S_SE_noisy = np.sqrt(real_SE**2 + imag_SE**2)
    S_IR_noisy = S_IR_t + np.random.normal(0, sigma, S_IR_t.shape)

    interpolator = create_interpolator(f_grid, t_grid)
    return interpolator(S_IR_noisy / S_SE_noisy).astype(np.single)


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
    logger.debug("Generating DICOM standard affine matrix from frame functional group metadata.")
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
    data, affine = data_reorientation(volume, C @ A_dcm)
    return MRIData(data=data, affine=affine)


def compute_mixed_t1_array(
    se_data: np.ndarray, ir_data: np.ndarray, meta: MetaDict, t1_low: float = 500.0, t1_high: float = 5000.0
) -> np.ndarray:
    """
    Computes a Mixed T1 array from Spin-Echo and Inversion-Recovery volumes using a lookup table.

    Args:
        se_data (np.ndarray): 3D numpy array of the Spin-Echo modulus data.
        ir_data (np.ndarray): 3D numpy array of the Inversion-Recovery corrected real data.
        meta (MetaDict): Dictionary containing sequence parameters ('TR_SE', 'TI', 'TE', 'ETL').
        t1_low (float): Lower bound for T1 generation grid. Defaults to 500 ms.
        t1_high (float): Upper bound for T1 generation grid. Defaults to 5000 ms.

    Returns:
        np.ndarray: Computed T1 map as a 3D float32 array.
    """
    logger.info("Computing Mixed T1 array from SE and IR data using lookup table interpolation.")
    nonzero_mask = se_data != 0
    f_data = np.nan * np.zeros_like(ir_data)
    f_data[nonzero_mask] = ir_data[nonzero_mask] / se_data[nonzero_mask]

    tr_se, ti, te, etl = meta["TR_SE"], meta["TI"], meta["TE"], meta["ETL"]
    tr_free = estimate_se_free_relaxation_time(tr_se, te, etl)

    t1_grid = np.arange(int(t1_low), int(t1_high + 1))
    S_SE, S_IR = T1_to_mixed_signals(t1_grid, TR=tr_free, TI=ti)
    f_curve = S_IR / S_SE

    logger.debug(
        f"Generated T1 lookup table with TR_SE={tr_se}, TI={ti}, TE={te}, "
        f"ETL={etl}, T1 range=({t1_low}, {t1_high}), table size={len(t1_grid)}"
    )
    interpolator = create_interpolator(f_curve, t1_grid)
    return interpolator(f_data).astype(np.single)


def create_interpolator(f_grid, t1_grid):
    interpolator = scipy.interpolate.interp1d(f_grid, t1_grid, kind="nearest", bounds_error=False, fill_value=np.nan)
    logger.debug("Created interpolation function for T1 estimation based on the lookup table.")
    return interpolator


def _extract_frame_metadata(frame_fg) -> dict:
    """
    Extracts core physical parameters (TR, TE, TI, ETL) from a DICOM frame functional group.

    Args:
        frame_fg: The PerFrameFunctionalGroupsSequence element for a specific frame.

    Returns:
        dict: A dictionary containing available MR timing parameters.
    """
    logger.debug("Extracting MR timing parameters from DICOM frame functional group.")
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
    logger.debug(f"Extracting subvolumes {subvolumes} from DICOM file {dcmpath}")
    import pydicom

    dcm = pydicom.dcmread(str(dcmpath))
    frames_total = int(dcm.NumberOfFrames)
    logger.debug(f"Total frames in DICOM: {frames_total}")
    # [0x2001, 0x1018] is a private Philips tag representing 'Number of Slices MR'
    frames_per_volume = dcm[0x2001, 0x1018].value
    num_volumes = frames_total // frames_per_volume
    assert num_volumes * frames_per_volume == frames_total, "Subvolume dimensions do not evenly divide the total frames."

    logger.debug(f"Frames per volume: {frames_per_volume}, Number of volumes: {num_volumes}")
    pixel_data = dcm.pixel_array.astype(np.single)
    frame_fg_sequence = dcm.PerFrameFunctionalGroupsSequence

    vols_out = []
    for volname in subvolumes:
        logger.debug(f"Processing subvolume '{volname}'")
        vol_idx = VOLUME_LABELS.index(volname)

        # Find volume slices representing the current subvolume
        subvol_idx_start = vol_idx * frames_per_volume
        subvol_idx_end = (vol_idx + 1) * frames_per_volume
        frame_fg = frame_fg_sequence[subvol_idx_start]

        logger.debug(
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


def dicom_to_mixed(dcmpath: Path, outpath: Path, subvolumes: list[str] | None = None):
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
    logger.info(f"Starting DICOM to Mixed conversion for {dcmpath} with output base {outpath}")

    subvolumes = subvolumes or VOLUME_LABELS
    logger.debug(f"Subvolumes to extract: {subvolumes}")
    assert all([volname in VOLUME_LABELS for volname in subvolumes]), (
        f"Invalid subvolume name in {subvolumes}, must be one of {VOLUME_LABELS}"
    )

    outdir, form = outpath.parent, outpath.stem
    logger.debug(f"Output directory: {outdir}, output form prefix: {form}")
    outdir.mkdir(exist_ok=True, parents=True)

    vols = extract_mixed_dicom(dcmpath, subvolumes)
    logger.debug(f"Extracted {len(vols)} subvolumes from DICOM, preparing to save NIfTI files and metadata.")
    meta = {}

    for vol, volname in zip(vols, subvolumes):
        output = outpath.with_name(f"{outpath.stem}_{volname}.nii.gz")
        logger.debug(f"Saving subvolume '{volname}' to {output}")
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
    json_meta_path = outdir / f"{form}_meta.json"
    logger.debug(f"Writing metadata JSON sidecar to {json_meta_path} with contents: {meta}")
    json_meta_path.write_text(json.dumps(meta, indent=4))

    # Attempt standard dcm2niix conversion (soft failure allowed for legacy behavior)
    logger.debug("Attempting to run dcm2niix for standard conversion (soft failure allowed).")
    run_dcm2niix(dcmpath, outdir, form, extra_args="-w 0 --terse -b o", check=False)


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    subparser = parser.add_subparsers(dest="hybrid-command", required=True, title="hybrid subcommands")

    dmc_parser = subparser.add_parser(
        "dcm2mixed",
        help="Convert a Mixed sequence DICOM file into separate NIfTI subvolumes and metadata.",
        formatter_class=parser.formatter_class,
    )
    dmc_parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input Mixed DICOM file.")
    dmc_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Base path for output NIfTI files and metadata JSON."
    )
    dmc_parser.add_argument(
        "-s",
        "--subvolumes",
        nargs="+",
        default=VOLUME_LABELS,
        help=f"Specific subvolumes to extract, space-separated. Defaults to all: {VOLUME_LABELS}.",
    )

    t1_parser = subparser.add_parser(
        "t1", help="Generate a T1 map from Mixed sequence NIfTI files.", formatter_class=parser.formatter_class
    )
    t1_parser.add_argument("-s", "--se", type=Path, required=True, help="Path to the Spin-Echo modulus NIfTI file.")
    t1_parser.add_argument(
        "-i", "--ir", type=Path, required=True, help="Path to the Inversion-Recovery corrected real NIfTI file."
    )
    t1_parser.add_argument(
        "-m", "--meta", type=Path, required=True, help="Path to the JSON file containing the sequence parameters."
    )
    t1_parser.add_argument("--t1-low", type=float, default=500.0, help="Lower bound for T1 interpolation grid (ms).")
    t1_parser.add_argument("--t1-high", type=float, default=5000.0, help="Upper bound for T1 interpolation grid (ms).")
    t1_parser.add_argument("-o", "--output", type=Path, required=True, help="Output path for the generated T1 map NIfTI file.")

    post_parser = subparser.add_parser(
        "postprocess",
        help="Mask a Mixed T1 map to isolate the CSF using the original SE sequence.",
        formatter_class=parser.formatter_class,
    )
    post_parser.add_argument(
        "-s", "--se", type=Path, required=True, help="Path to the Spin-Echo modulus NIfTI file used to derive the mask."
    )
    post_parser.add_argument(
        "-t", "--t1", type=Path, required=True, help="Path to the previously generated Mixed T1 map NIfTI file."
    )
    post_parser.add_argument("-o", "--output", type=Path, required=True, help="Output path for the masked T1 map NIfTI file.")

    if extra_args_cb is not None:
        extra_args_cb(dmc_parser)
        extra_args_cb(t1_parser)
        extra_args_cb(post_parser)


def dispatch(args):
    """Dispatch function for the mixed T1 map generation commands."""
    command = args.pop("hybrid-command")  # Note: matches the 'dest' in your add_arguments

    if command == "dcm2mixed":
        dicom_to_mixed(dcmpath=args.pop("input"), outpath=args.pop("output"), subvolumes=args.pop("subvolumes"))
    elif command == "t1":
        nii = Mixed.from_file(
            se_path=args.pop("se"),
            ir_path=args.pop("ir"),
            meta_path=args.pop("meta"),
        ).t1_map(
            T1_low=args.pop("t1_low"),
            T1_high=args.pop("t1_high"),
        )
        output = args.pop("output")
        if output is not None:
            nii.save(output)
            logger.info(f"Saved Mixed T1 map to {output}")

    elif command == "postprocess":
        nii = MixedT1.from_file(
            se_path=args.pop("se"),
            t1_path=args.pop("t1"),
        ).postprocess()
        output = args.pop("output")
        if output is not None:
            nii.save(output)
            logger.info(f"Saved masked Mixed T1 map to {output}")
    else:
        raise ValueError(f"Unknown command: {command}")
