# MRI Segmentation - Lookup Table (LUT) Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

import argparse
import itertools
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from .data import MRIData, apply_affine
from .testing import assert_same_space

logger = logging.getLogger(__name__)


# Regex to match a standard FreeSurfer Color LUT record line
LUT_REGEX = re.compile(r"^(?P<label>\d+)\s+(?P<description>[_\da-zA-Z-]+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+(?P<A>\d+)")
CORPUS_CALLOSUM = [251, 252, 253, 254, 255]

CEREBRAL_WM_RANGES = [
    *[2, 41],  # aseg left/right cerebral white labels
    *list(range(3000, 3036)),  # wmparc-left-labels
    *list(range(4000, 4036)),  # wmparc-right-labels
    *[5001, 5002],
    *[28, 60],  # VentralDC included in white matter subdomain
    *CORPUS_CALLOSUM,  # Corpus callosum
    *[31, 63],  # Choroid plexus.
]

CEREBRAL_CGM_RANGES = [
    *[3, 42],  # aseg left/right cortical gm
    *list(range(1000, 1036)),  # aparc left labels
    *list(range(2000, 2036)),  # aparc right labels
]

SUBCORTICAL_GM_RANGES = [
    *(10, 49),  # Thalamus,
    *(11, 50),  # Caudate,
    *(12, 51),  # Putamen,
    *(13, 52),  # pallidum
    *(17, 53),  # hippocampus
    *(18, 54),  # amygdala
    *(26, 58),  # accumbens
]

VENTRICLES = [
    4,  # Left lateral ventricle
    5,  # Left inferior lateral ventricle
    14,  # Third ventricle
    15,  # Fourth ventricle
    43,  # Right lateral ventricle
    44,  # Right inferior lateral ventricle
]

FREESURFER_CSF = [
    *VENTRICLES,
    24,  # Generic CSF
]

CORTICAL_CSF = [
    *(x + 15000 for x in CEREBRAL_CGM_RANGES),
]

SEGMENTATION_GROUPS = {
    "cerebral-wm": CEREBRAL_WM_RANGES,
    "cerebral-cortex": CEREBRAL_CGM_RANGES,
    "cerebellar-wm": [7, 46],  # left, right
    "cerebellar-cortex": [8, 47],  # left, right
    "csf-freesurfer": FREESURFER_CSF,
    "cortical-csf": CORTICAL_CSF,
    "corpus-callosum": CORPUS_CALLOSUM,
    "subcortical-gm": SUBCORTICAL_GM_RANGES,
}


class Segmentation:
    """
    Base class for MRI segmentations, linking spatial data with anatomical lookup tables.

    This class extends MRIData by specifically treating the image array as discrete
    integer labels representing Regions of Interest (ROIs). It links these numerical
    labels to a descriptive Lookup Table (LUT).
    """

    mri: MRIData
    rois: np.ndarray
    lut: pd.DataFrame
    label_name: str

    def __init__(self, mri: MRIData, lut: pd.DataFrame | None = None):
        """
        Initializes the Segmentation object.

        Args:
            data (np.ndarray): 3D numpy array containing integer ROI labels.
            affine (np.ndarray): 4x4 affine transformation matrix mapping voxel indices to physical space.
            lut (Optional[pd.DataFrame], optional): A pandas DataFrame mapping numerical labels
                to their descriptions. If None, a default numerical mapping is generated. Defaults to None.
        """
        self.mri = mri

        # Extract all unique active regions (ignoring 0/background)
        self.rois = np.unique(self.mri.data[self.mri.data > 0])

        if lut is not None:
            self.lut = lut
        else:
            self.lut = pd.DataFrame({"Label": self.rois}, index=self.rois)

        # Identify the primary label column dynamically
        self.label_name = "Label" if "Label" in self.lut.columns else self.lut.columns[0]

    @classmethod
    def from_file(
        cls, seg_path: Path, dtype: npt.DTypeLike | None = None, orient: bool = True, lut_path: Path | None = None
    ) -> "Segmentation":
        """Loads a Segmentation from a NIfTI file.

        Args:
              seg_path (Path): The file path to the segmentation NIfTI file.
              dtype (npt.DTypeLike, optional): The data type for the segmentation data. Defaults to None.
              orient (bool, optional): Whether to orient the data. Defaults to True.
              lut_path (Path, optional): The file path to the lookup table. Defaults to None.
          Returns:
              Segmentation: An instance of the Segmentation class containing the loaded
              segmentation data and affine transformation.
        """
        logger.info(f"Loading segmentation from {seg_path}.")
        mri = MRIData.from_file(seg_path, dtype=dtype, orient=orient)

        if lut_path is None and seg_path.with_suffix(".json").exists():
            lut_path = seg_path.with_suffix(".json")

        if lut_path is not None:
            logger.info(f"Loading LUT from {lut_path}.")
            lut = pd.read_json(lut_path)
        else:
            rois = np.unique(mri.data[mri.data > 0])
            lut = pd.DataFrame({"Label": rois}, index=rois)

        return cls(mri=mri, lut=lut)

    def save(self, output_path: Path, dtype: npt.DTypeLike | None = None, intent_code: int = 1006, lut_path: Path | None = None):
        """Saves the Segmentation to a NIfTI file.

        Args:
            output_path (Path): The file path where the segmentation will be saved.
            dtype (npt.DTypeLike, optional): The data type for the saved segmentation data. Defaults to None.
            intent_code (int, optional): The NIfTI intent code to set in the header. Defaults to 1006 (NIFTI_INTENT_LABEL).
        """
        self.mri.save(output_path, dtype=dtype, intent_code=intent_code)
        if lut_path is not None:
            self.lut.to_json(lut_path, orient="index")
        else:
            self.lut.to_json(output_path.with_suffix(".json"), orient="index")

    def set_lut(self, lut: pd.DataFrame, label_column: str = "Label"):
        """Sets the Lookup Table (LUT) for the segmentation, ensuring it matches the present ROIs.

        Args:
            lut (pd.DataFrame): A pandas DataFrame mapping numerical labels
                to their descriptions. If None, a default numerical mapping is generated. Defaults to None.
            label_column (str, optional): The name of the column in the LUT that contains the label
                descriptions. Defaults to "Label".
        """

        self.lut = lut
        self.label_name = label_column
        if self.label_name not in self.lut.columns:
            raise ValueError(f"Specified label column '{self.label_name}' not found in LUT.")

    @property
    def num_rois(self) -> int:
        """The number of unique active regions of interest present
        in the segmentation volume.
        """
        return len(self.rois)

    @property
    def roi_labels(self) -> np.ndarray:
        """An array containing the unique numerical labels of all present ROIs."""
        return self.rois

    def get_roi_labels(self, rois: npt.NDArray[np.int32] | None = None) -> pd.DataFrame:
        """
        Retrieves a descriptive mapping for a specified set of ROIs.

        Args:
            rois (Optional[npt.NDArray[np.int32]], optional): Array of numerical ROIs to look up.
                If None, retrieves labels for all ROIs currently present in the volume. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with columns ['ROI', <label_name>] mapping the numbers
            to their string descriptions.

        Raises:
            ValueError: If any requested ROI is not present in the segmentation volume.
        """
        if rois is None:
            rois = self.rois

        if not np.isin(rois, self.rois).all():
            raise ValueError("Some of the provided ROIs are not present in the segmentation.")

        return self.lut.loc[self.lut.index.isin(rois), [self.label_name]].rename_axis("ROI").reset_index()

    def resample_to_reference(self, reference_mri: MRIData) -> "Segmentation":
        """
        Resamples the segmentation to match the spatial dimensions and resolution of a reference MRI.

        Args:
            reference_mri (MRIData): The MRI to which the segmentation will be resampled,
                                     for example a T1-weighted anatomical scan.
        Returns:
            Segmentation: A new Segmentation object containing the resampled data.
        """

        shape_in = self.mri.shape
        shape_out = reference_mri.shape

        # Generate a grid of voxel indices for the output space
        upsampled_indices = np.fromiter(
            itertools.product(*(np.arange(ni) for ni in shape_out)),
            dtype=np.dtype((int, 3)),
        )
        # Get voxel indices in the input segmentation space corresponding to the output grid
        seg_indices = apply_affine(
            np.linalg.inv(self.mri.affine),
            apply_affine(reference_mri.affine, upsampled_indices),
        )
        seg_indices = np.rint(seg_indices).astype(int)

        # The two images does not necessarily share field of view.
        # Remove voxels which are not located within the segmentation fov.
        valid_index_mask = (seg_indices > 0).all(axis=1) * (seg_indices < shape_in).all(axis=1)
        upsampled_indices = upsampled_indices[valid_index_mask]
        seg_indices = seg_indices[valid_index_mask]

        seg_upsampled = np.zeros(shape_out, dtype=self.mri.data.dtype)
        I_in, J_in, K_in = seg_indices.T
        I_out, J_out, K_out = upsampled_indices.T
        seg_upsampled[I_out, J_out, K_out] = self.mri.data[I_in, J_in, K_in]

        # return Segmentation(data=seg_upsampled, affine=reference_mri.affine, lut=self.lut)
        mri = MRIData(data=seg_upsampled, affine=reference_mri.affine)
        return Segmentation(mri=mri, lut=self.lut)

    def smooth(self, sigma: float, cutoff_score: float = 0.5, **kwargs) -> "Segmentation":
        """
        Applies Gaussian smoothing to the segmentation labels to create a soft probabilistic map.

        Args:
            sigma (float): The standard deviation for the Gaussian kernel.
            cutoff_score (float, optional): A threshold to remove low-confidence voxels. Defaults to 0.5.
            **kwargs: Additional keyword arguments passed to scipy.ndimage.gaussian_filter.

        Returns:
            dict[str, np.ndarray]: A dictionary containing 'labels' (the smoothed segmentation)
            and 'scores' (the confidence scores for each voxel).
        """
        smoothed_rois = np.zeros_like(self.mri.data)
        high_scores = np.zeros(self.mri.data.shape)

        for roi in self.rois:
            scores = scipy.ndimage.gaussian_filter((self.mri.data == roi).astype(float), sigma=sigma, **kwargs)
            is_new_high_score = scores > high_scores
            smoothed_rois[is_new_high_score] = roi
            high_scores[is_new_high_score] = scores[is_new_high_score]

        delete_scores = (high_scores < cutoff_score) * (self.mri.data == 0)
        smoothed_rois[delete_scores] = 0

        mri = MRIData(data=smoothed_rois, affine=self.mri.affine)
        return Segmentation(mri=mri, lut=self.lut)


class FreeSurferSegmentation(Segmentation):
    """
    Segmentation class specifically tailored for FreeSurfer outputs.

    Automatically handles the loading of NIfTI files and the resolution of the
    standard FreeSurfer Color Lookup Table.
    """

    @classmethod
    def from_file(
        cls, filepath: Path | str, dtype: npt.DTypeLike | None = None, orient: bool = True, lut_path: Path | None = None
    ) -> "FreeSurferSegmentation":
        """
        Load a FreeSurfer segmentation from a NIfTI file, automatically resolving the LUT.

        Args:
            filepath (Path | str): Path to the NIfTI segmentation file.
            dtype (Optional[npt.DTypeLike], optional): Requested data type. Defaults to None.
            orient (bool, optional): Whether to reorient the data to standard space. Defaults to True.
            lut_path (Optional[Path], optional): Path to a custom FreeSurfer Color LUT.
                If None, standard fallback paths are checked. Defaults to None.

        Returns:
            FreeSurferSegmentation: The initialized segmentation object with the attached LUT.
        """
        resolved_lut_path = resolve_freesurfer_lut_path(lut_path)
        lut = read_freesurfer_lut(resolved_lut_path)

        # FreeSurfer LUTs index by the "label" column
        lut = lut.set_index("label") if "label" in lut.columns else lut

        mri = MRIData.from_file(filepath, dtype=dtype, orient=orient)
        return cls(mri=mri, lut=lut)


class ExtendedFreeSurferSegmentation(FreeSurferSegmentation):
    """
    Extended FreeSurfer segmentation handling custom tissue type classifications.

    Supports segmentation conventions where regions are offset by multiples of 10000
    to indicate broad tissue categories (e.g., Parenchyma, CSF, Dura) while preserving
    the base FreeSurfer anatomical label (modulus 10000).
    """

    def get_roi_labels(self, rois: npt.NDArray[np.int32] | None = None) -> pd.DataFrame:
        """
        Retrieves descriptive mappings including the augmented tissue type classifications.

        Args:
            rois (Optional[npt.NDArray[np.int32]], optional): Array of numerical ROIs to look up.
                If None, retrieves labels for all ROIs currently present. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame mapping the requested ROIs to their base descriptions
            and their computed 'tissue_type'.
        """
        rois = self.rois if rois is None else rois

        # Use modulus 10000 to extract the base anatomical label from the superclass LUT
        freesurfer_labels = super().get_roi_labels(rois % 10000).rename(columns={"ROI": "FreeSurfer_ROI"})

        # Get the broad tissue categories based on the numerical offsets
        tissue_type = self.get_tissue_type(rois)

        # Merge the base anatomical names with the tissue types
        return freesurfer_labels.merge(
            tissue_type,
            left_on="FreeSurfer_ROI",
            right_on="FreeSurfer_ROI",
            how="outer",
        ).drop(columns=["FreeSurfer_ROI"])[["ROI", self.label_name, "tissue_type"]]

    def get_tissue_type(self, rois: npt.NDArray[np.int32] | None = None) -> pd.DataFrame:
        """
        Determines the tissue type based on the numerical ranges of the ROI labels.

        Labels < 10000 are classified as "Parenchyma".
        Labels < 20000 are classified as "CSF".
        Labels >= 20000 are classified as "Dura".

        Args:
            rois (Optional[npt.NDArray[np.int32]], optional): Array of numerical ROIs to evaluate.
                If None, evaluates all ROIs currently present. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame mapping original ROIs to 'FreeSurfer_ROI' (modulus 10000)
            and their 'tissue_type'.
        """
        rois = self.rois if rois is None else rois

        tissue_types = pd.Series(
            data=np.where(rois < 10000, "Parenchyma", np.where(rois < 20000, "CSF", "Dura")),
            index=rois,
            name="tissue_type",
        )

        ret = pd.DataFrame(tissue_types, columns=["tissue_type"]).rename_axis("ROI").reset_index()
        ret["FreeSurfer_ROI"] = ret["ROI"] % 10000
        return ret


# @dataclass
class CSFSegmentation:
    segmentation: MRIData
    csf_mask: MRIData

    def __init__(self, segmentation: MRIData, csf_mask: MRIData):
        assert_same_space(segmentation, csf_mask)
        self.segmentation = segmentation
        self.csf_mask = csf_mask

    @classmethod
    def from_file(cls, segmentation_path: Path, csf_mask_path: Path) -> "CSFSegmentation":
        segmentation = MRIData.from_file(segmentation_path, dtype=np.int16)
        csf_mask = MRIData.from_file(csf_mask_path, dtype=bool)
        assert_same_space(segmentation, csf_mask)
        return cls(segmentation=segmentation, csf_mask=csf_mask)

    def to_csf_segmentation(self) -> MRIData:
        # Get interpolation operator
        I, J, K = np.where(self.segmentation.data != 0)
        interp = scipy.interpolate.NearestNDInterpolator(np.array([I, J, K]).T, self.segmentation.data[I, J, K])
        # Interpolate segmentation values at CSF mask locations
        i, j, k = np.where(self.csf_mask.data != 0)
        csf_seg = np.zeros_like(self.segmentation.data, dtype=np.int16)
        csf_seg[i, j, k] = interp(i, j, k)

        return MRIData(data=csf_seg.astype(np.int16), affine=self.csf_mask.affine)


def default_segmentation_groups() -> dict[str, list[int]]:
    """
    Returns the default grouping of FreeSurfer labels into brain regions.

    Returns:
        dict[str, list[int]]: A dictionary mapping human-readable anatomical
        region names (e.g., 'cerebral-wm') to a list of integer labels
        corresponding to those regions in the FreeSurfer Color LUT.
    """
    return {**SEGMENTATION_GROUPS}


def lut_record(match: re.Match) -> dict[str, str | float | int]:
    """
    Parses a regular expression match of a LUT line into a formatted dictionary.

    Normalizes RGB values from [0, 255] to [0.0, 1.0].
    Inverts the Alpha channel (FreeSurfer uses A=0 for opaque and A=255 for transparent,
    whereas standard rendering uses A=1.0 for opaque and A=0.0 for transparent).

    Args:
        match (re.Match): A regex match object containing label, description, R, G, B, and A groups.

    Returns:
        dict[str, str | float | int]: A parsed dictionary representing the LUT record.
    """
    groupdict = match.groupdict()
    return {
        "label": int(groupdict["label"]),
        "description": groupdict["description"],
        "R": float(groupdict["R"]) / 255.0,
        "G": float(groupdict["G"]) / 255.0,
        "B": float(groupdict["B"]) / 255.0,
        "A": 1.0 - float(groupdict["A"]) / 255.0,
    }


def validate_lut_file(filepath: Path) -> bool:
    """
    Validates that a file exists and contains valid FreeSurfer LUT records.

    Instead of a strict checksum (which would break custom LUTs), this
    checks if the file contains at least one valid record line within
    its first 50 lines.

    Args:
        filepath (Path): Path to the file to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                if LUT_REGEX.match(line):
                    return True
    except (IOError, UnicodeDecodeError):
        return False

    return False


def resolve_freesurfer_lut_path(filename: Path | None = None) -> Path:
    """
    Resolves and validates the file path for a Color Lookup Table in FreeSurfer.

    If a filename is provided, it validates it. If it doesn't exist, it uses
    that path as the download target. If no filename is provided, it defaults
    to 'FreeSurferColorLUT.txt' in FREESURFER_HOME or the current directory.
    Missing or invalid files trigger a fresh download from the official repository.

    Args:
        filename (Path | None, optional): Requested path to the LUT file. Defaults to None.

    Returns:
        Path: The resolved and validated absolute path to the LUT file.

    Raises:
        ValueError: If a provided file exists but has invalid formatting.
        RuntimeError: If the downloaded fallback file is invalid.
    """
    if filename is not None:
        target_path = Path(filename)
        if target_path.exists():
            if validate_lut_file(target_path):
                return target_path
            else:
                raise ValueError(f"Provided LUT file '{target_path}' is invalid or corrupted.")
    else:
        # Default fallback target if none provided
        target_name = "FreeSurferColorLUT.txt"
        filedir = Path(os.environ.get("FREESURFER_HOME", Path.cwd()))
        target_path = filedir / target_name

    # If the target path doesn't exist (whether custom or default) or is invalid, download it.
    if not target_path.exists() or not validate_lut_file(target_path):
        logger.info(f"Valid LUT file not found. Downloading to {target_path}...")

        # Ensure the parent directory exists before downloading
        target_path.parent.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/freesurfer/freesurfer/raw/dev/distribution/FreeSurferColorLUT.txt"
        urlretrieve(url, target_path)

        if not validate_lut_file(target_path):
            raise RuntimeError(f"Downloaded LUT file at '{target_path}' is invalid. Download may have failed.")

    return target_path


def read_freesurfer_lut(filename: Path | None = None) -> pd.DataFrame:
    """
    Reads a FreeSurfer Color Lookup Table text file into a Pandas DataFrame.

    Missing files will be automatically downloaded. RGBA values are normalized to floats [0.0, 1.0].

    Args:
        filename (Path | None, optional): Path to the LUT text file. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns ['label', 'description', 'R', 'G', 'B', 'A'].
    """
    resolved_path = resolve_freesurfer_lut_path(filename)

    with open(resolved_path, "r", encoding="utf-8") as f:
        records = [lut_record(m) for m in map(LUT_REGEX.match, f) if m is not None]

    return pd.DataFrame.from_records(records)


def write_lut(filename: Path, table: pd.DataFrame):
    """
    Writes a Pandas DataFrame back to the FreeSurfer Color Lookup Table text format.

    Reverses the normalization applied during `read_freesurfer_lut`, converting float [0.0, 1.0] RGBA
    values back to integer [0, 255] values, and re-inverting the Alpha channel.

    Args:
        filename (Path): Path where the LUT file will be saved.
        table (pd.DataFrame): The DataFrame containing the LUT records.
    """
    newtable = table.copy()

    # Re-scale RGB values to [0, 255]
    for col in ["R", "G", "B"]:
        newtable[col] = (newtable[col] * 255).astype(int)

    # Reverse Alpha inversion and scale to [0, 255]
    newtable["A"] = 255 - (newtable["A"] * 255).astype(int)

    # Save as tab-separated values without headers or indices
    newtable.to_csv(filename, sep="\t", index=False, header=False)


def add_arguments(
    parser: argparse.ArgumentParser,
    extra_args_cb: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    subparser = parser.add_subparsers(dest="seg-command", help="Commands for segmentation processing")

    resample_parser = subparser.add_parser(
        "resample", help="Resample a segmentation to match the space of a reference MRI", formatter_class=parser.formatter_class
    )
    resample_parser.add_argument("-i", "--input", type=Path, help="Path to the input segmentation NIfTI file")
    resample_parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        help="Path to the reference MRI \
        - usually a registered T1 weighted anatomical scan",
    )
    resample_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the resampled segmentation")

    smooth_parser = subparser.add_parser(
        "smooth",
        help="Apply Gaussian smoothing to a segmentation to create a soft probabilistic map",
        formatter_class=parser.formatter_class,
    )
    smooth_parser.add_argument("-i", "--input", type=Path, help="Path to the input (refined) segmentation NIfTI file")
    smooth_parser.add_argument("-s", "--sigma", type=float, help="Standard deviation for the Gaussian kernel used in smoothing")
    smooth_parser.add_argument(
        "-c", "--cutoff", type=float, default=0.5, help="Cutoff score to remove low-confidence voxels (default: 0.5)"
    )
    smooth_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the smoothed segmentation")

    refine_parser = subparser.add_parser(
        "refine",
        help="Refine a segmentation by applying Gaussian smoothing to the labels",
        formatter_class=parser.formatter_class,
    )
    refine_parser.add_argument("-i", "--input", type=Path, help="Path to the input segmentation NIfTI file")
    refine_parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        help="Path to the reference MRI \
        - usually a registered T1 weighted anatomical scan",
    )
    refine_parser.add_argument("-s", "--smooth", type=float, help="Standard deviation for the Gaussian kernel used in smoothing")
    refine_parser.add_argument("-o", "--output", type=Path, help="Desired output path for the refined segmentation")

    if extra_args_cb is not None:
        extra_args_cb(resample_parser)
        extra_args_cb(smooth_parser)
        extra_args_cb(refine_parser)


def dispatch(args):
    command = args.pop("seg-command")
    if command == "resample":
        print("Resampling segmentation...")
        input_seg = Segmentation.from_file(args.pop("input"))
        reference_mri = MRIData.from_file(args.pop("reference"))
        resampled_seg = input_seg.resample_to_reference(reference_mri)
        resampled_seg.save(args.pop("output"), dtype=np.int32)

    elif command == "smooth":
        smoothed = Segmentation.from_file(args.pop("input")).smooth(sigma=args.pop("sigma"), cutoff_score=args.pop("cutoff"))
        smoothed.save(args.pop("output"), dtype=np.int32)

    elif command == "refine":
        seg = Segmentation.from_file(args.pop("input"))
        refined = seg.resample_to_reference(MRIData.from_file(args.pop("reference")))
        smoothed = refined.smooth(sigma=args.pop("smooth"))
        refined.mri.data = np.where(smoothed.mri.data > 0, smoothed.mri.data, refined.mri.data)
        refined.save(args.pop("output"), dtype=np.int32)

    else:
        raise ValueError(f"Unknown segmentation command: {command}")
