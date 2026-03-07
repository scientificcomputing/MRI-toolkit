import numpy as np
from pathlib import Path

from unittest.mock import MagicMock, patch

from mritk.t1.mixed import (
    compute_mixed_t1_array,
    extract_mixed_dicom,
    mixed_t1map,
    mixed_t1map_postprocessing,
    _extract_frame_metadata,
)
from mritk.t1.utils import VOLUME_LABELS, compare_nifti_images


def test_mixed_t1map(tmp_path, mri_data_dir: Path):
    SE_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_SE-modulus.nii.gz"
    IR_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_IR-corrected-real.nii.gz"
    meta_path = mri_data_dir / "mri-dataset/mri_dataset/sub-01" / "ses-01/mixed/sub-01_ses-01_acq-mixed_meta.json"

    ref_output = mri_data_dir / "mri-processed/mri_dataset/derivatives/sub-01" / "ses-01/sub-01_ses-01_acq-mixed_T1map.nii.gz"
    test_output_raw = tmp_path / "output_acq-mixed_T1map_raw.nii.gz"
    test_output = tmp_path / "output_acq-mixed_T1map.nii.gz"

    T1_low = 100
    T1_high = 10000

    mixed_t1map(
        SE_nii_path=SE_path, IR_nii_path=IR_path, meta_path=meta_path, T1_low=T1_low, T1_high=T1_high, output=test_output_raw
    )
    mixed_t1map_postprocessing(SE_nii_path=SE_path, T1_path=test_output_raw, output=test_output)

    compare_nifti_images(test_output, ref_output, data_tolerance=1e-12)


def test_compute_mixed_t1_array():
    """Test generating a T1 map from SE and IR modalities via interpolation."""
    se_data = np.array([[[1000.0, 1000.0]]])
    # IR signals at varying levels
    ir_data = np.array([[[-500.0, 500.0]]])

    meta = {"TR_SE": 1000.0, "TI": 100.0, "TE": 10.0, "ETL": 5}

    t1_low = 100.0
    t1_high = 3000.0

    t1_volume = compute_mixed_t1_array(se_data, ir_data, meta, t1_low, t1_high)

    # Should output same shape
    assert t1_volume.shape == (1, 1, 2)
    # T1 maps should not contain negative values in valid tissue
    assert np.all(t1_volume[~np.isnan(t1_volume)] > 0)


def test_extract_frame_metadata():
    """Test the extraction of relevant MR metadata parameters from DICOM tags."""
    # Mocking a DICOM Functional Group hierarchy
    mock_frame = MagicMock()
    mock_frame.MRTimingAndRelatedParametersSequence[0].RepetitionTime = 1500.0
    mock_frame.MREchoSequence[0].EffectiveEchoTime = 10.0
    mock_frame.MRModifierSequence[0].InversionTimes = [150.0]
    mock_frame.MRTimingAndRelatedParametersSequence[0].EchoTrainLength = 5

    meta = _extract_frame_metadata(mock_frame)

    assert meta["TR"] == 1500.0
    assert meta["TE"] == 10.0
    assert meta["TI"] == 150.0
    assert meta["ETL"] == 5


@patch("mritk.t1.mixed.extract_single_volume")
@patch("pydicom.dcmread")
def test_extract_mixed_dicom(mock_dcmread, mock_extract_single):
    """Test parsing a multi-volume DICOM file into independent subvolumes."""
    # Mocking the pydicom output
    mock_dcm = MagicMock()
    mock_dcm.NumberOfFrames = 20
    # Private tag for "Number of slices MR"
    mock_slice_tag = MagicMock()
    mock_slice_tag.value = 10

    # We have to mock __getitem__ because it's called via dcm[0x2001, 0x1018]
    def getitem_side_effect(key):
        if key == (0x2001, 0x1018):
            return mock_slice_tag
        return MagicMock()

    mock_dcm.__getitem__.side_effect = getitem_side_effect

    # Dummy pixel array
    mock_dcm.pixel_array = np.zeros((20, 2, 2))

    # Mocking Frame metadata sequences
    mock_frame_fg = MagicMock()
    mock_frame_fg.MRTimingAndRelatedParametersSequence[0].RepetitionTime = 1000.0
    mock_frame_fg.MREchoSequence[0].EffectiveEchoTime = 5.0

    # List of 20 frames
    mock_dcm.PerFrameFunctionalGroupsSequence = [mock_frame_fg] * 20
    mock_dcmread.return_value = mock_dcm

    # Mock the volume extraction output
    mock_mri_data = MagicMock()
    mock_mri_data.data = np.ones((10, 2, 2))
    mock_mri_data.affine = np.eye(4)
    mock_extract_single.return_value = mock_mri_data

    # Run the function requesting just the first two volumes
    dcmpath = Path("/dummy/file.dcm")
    test_subvolumes = [VOLUME_LABELS[0], VOLUME_LABELS[1]]

    results = extract_mixed_dicom(dcmpath, test_subvolumes)

    # Verifications
    assert len(results) == 2
    assert "nifti" in results[0]
    assert "descrip" in results[0]
    assert results[0]["descrip"]["TR"] == 1000.0

    # Ensure extract_single_volume was called twice (once for each subvolume)
    assert mock_extract_single.call_count == 2
