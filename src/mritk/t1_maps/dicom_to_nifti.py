"""MRI DICOM to NIfTI conversion Module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import nibabel
import json

import numpy as np

from ..data.io import load_mri_data, save_mri_data
from ..dicom.utils import VOLUME_LABELS, read_dicom_trigger_times


def extract_mixed_dicom(dcmpath: Path, subvolumes: list[str]):
    import pydicom

    dcm = pydicom.dcmread(dcmpath)
    frames_total = int(dcm.NumberOfFrames)
    frames_per_volume = dcm[0x2001, 0x1018].value  # [Number of Slices MR]
    num_volumes = frames_total // frames_per_volume
    assert num_volumes * frames_per_volume == frames_total, (
        "Subvolume dimensions do not match"
    )

    D = dcm.pixel_array.astype(np.single)
    frame_fg_sequence = dcm.PerFrameFunctionalGroupsSequence

    vols_out = []
    for volname in subvolumes:
        vol_idx = VOLUME_LABELS.index(volname)

        # Find volume slices representing current subvolume
        subvol_idx_start = vol_idx * frames_per_volume
        subvol_idx_end = (vol_idx + 1) * frames_per_volume
        frame_fg = frame_fg_sequence[subvol_idx_start]
        logger.info(
            (
                f"Converting volume {vol_idx + 1}/{len(VOLUME_LABELS)}: {volname} between indices"
                + f"{subvol_idx_start, subvol_idx_end} / {frames_total}."
            )
        )
        mri = extract_single_volume(D[subvol_idx_start:subvol_idx_end], frame_fg)

        nii_oriented = nibabel.nifti1.Nifti1Image(mri.data, mri.affine)
        nii_oriented.set_sform(nii_oriented.affine, "scanner")
        nii_oriented.set_qform(nii_oriented.affine, "scanner")

        # Include meta-data
        description = {
            "TR": float(
                frame_fg.MRTimingAndRelatedParametersSequence[0].RepetitionTime
            ),
            "TE": float(frame_fg.MREchoSequence[0].EffectiveEchoTime),
        }
        if hasattr(frame_fg.MRModifierSequence[0], "InversionTimes"):
            description["TI"] = frame_fg.MRModifierSequence[0].InversionTimes[0]
        if hasattr(frame_fg.MRTimingAndRelatedParametersSequence[0], "EchoTrainLength"):
            description["ETL"] = frame_fg.MRTimingAndRelatedParametersSequence[
                0
            ].EchoTrainLength
        vols_out.append({"nifti": nii_oriented, "descrip": description})
    return vols_out


def dicom_to_looklocker(
    dicomfile: Path,
    outpath: Path
):
    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)
    times = read_dicom_trigger_times(dicomfile)
    np.savetxt(f"{outdir}/{form}" + "_trigger_times.txt", times)

    with tempfile.TemporaryDirectory(prefix=outpath.stem) as tmpdir:
        tmppath = Path(tmpdir)
        cmd = f"dcm2niix -f {form} -z y --ignore_trigger_times -o '{tmppath}' '{dicomfile}' > /tmp/dcm2niix.txt"
        subprocess.run(cmd, shell=True, check=True)
        shutil.copy(
            tmppath / f"{form}.json",
            outpath.with_suffix(".json"),
        )
        mri = load_mri_data(
            tmppath / f"{form}.nii.gz", dtype=np.double
        )
        save_mri_data(
            mri, outpath.with_suffix(".nii.gz"), dtype=np.single, intent_code=2001
        )


def dicom_to_mixed(
    dcmpath: Path,
    outpath: Path,
    subvolumes: Optional[list[str]] = None,
):
    subvolumes = subvolumes or VOLUME_LABELS
    assert all([volname in VOLUME_LABELS for volname in subvolumes]), (
        f"Invalid subvolume name in {subvolumes}, not in {VOLUME_LABELS}"
    )
    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    vols = extract_mixed_dicom(dcmpath, subvolumes)
    meta = {}
    for vol, volname in zip(vols, subvolumes):
        output = outpath.with_name(outpath.stem + "_" + volname + ".nii.gz")

        nii = vol["nifti"]
        descrip = vol["descrip"]
        nibabel.nifti1.save(nii, output)
        try:
            if volname == "SE-modulus":
                meta["TR_SE"] = descrip["TR"]
                meta["TE"] = descrip["TE"]
                meta["ETL"] = descrip["ETL"]
            elif volname == "IR-corrected-real":
                meta["TR_IR"] = descrip["TR"]
                meta["TI"] = descrip["TI"]
        except KeyError as e:
            print(volname, descrip)
            raise e

    with open(outpath.parent / f"{form}_meta.json", "w") as f:
        json.dump(meta, f)

    try:
        cmd = f"dcm2niix -w 0 --terse -b o -f '{form}' -o '{outdir}' '{dcmpath}' >> /tmp/dcm2niix.txt "
        subprocess.run(cmd, shell=True).check_returncode()
    except (ValueError, subprocess.CalledProcessError) as e:
        print(str(e))
        pass