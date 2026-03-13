from ..data.base import MRIData
import numpy as np
import pandas as pd
from .lookup_table import read_lut
from pathlib import Path
from ..data.io import load_mri_data


class Segmentation(MRIData):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        super().__init__(data, affine)
        self.data = self.data.astype(int)
        self.rois = np.unique(self.data[self.data > 0])

    @classmethod
    def from_file(cls, filepath: Path, dtype: type = int):
        data, affine = load_mri_data(filepath, dtype=dtype)
        return cls(data=data, affine=affine)

    @property
    def num_rois(self) -> int:
        return len(self.rois)

    @property
    def roi_labels(self) -> np.ndarray:
        return self.rois


class FreeSurferSegmentation(Segmentation):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        super().__init__(data, affine)

        # TODO: verify that all labels in the segmentation are valid FS labels
        # Retrieve the FreeSurfer LUT, which contains the labels associated with each ROI
        self.freesurfer_lut = read_lut(None)  # Finds freesurfer LUT

    def get_roi_labels(self, rois: np.ndarray[int] | None = None) -> pd.DataFrame:
        if not np.isin(rois, self.rois).all():
            raise ValueError("Some of the provided ROIs are not present in the segmentation.")

        if rois is None:
            rois = self.rois
        return self.freesurfer_lut.loc[self.freesurfer_lut.index.isin(rois), ["Label"]].rename_axis("ROI").reset_index()


class ExtendedFreeSurferSegmentation(FreeSurferSegmentation):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        super().__init__(data, affine)

    def get_roi_labels(self, rois: np.ndarray[int] | None = None) -> pd.DataFrame:
        rois = self.rois if rois is None else rois

        return super().get_roi_labels(rois % 10000)
        # TODO: Add column to distinguish parenchyma, csf etc
