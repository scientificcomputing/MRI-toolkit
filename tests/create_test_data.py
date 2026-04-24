import zipfile
from pathlib import Path


def main():
    outdir = Path("mritk-test-data")
    inputdir = Path("gonzo")  # Assumes you have the Gonzo dataset downloaded here
    files = [
        "timetable/timetable.tsv",
        "mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1.nii.gz",
        "mri-dataset/mri_dataset/sub-01/ses-01/anat/sub-01_ses-01_acq-looklocker_IRT1_trigger_times.txt",
        "mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_SE-modulus.nii.gz",
        "mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_IR-corrected-real.nii.gz",
        "mri-dataset/mri_dataset/sub-01/ses-01/mixed/sub-01_ses-01_acq-mixed_meta.json",
        "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_acq-mixed_T1map_registered.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf_binary.nii.gz",
        "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-01_T1map_hybrid.nii.gz",
        "mri-processed/mri_processed_data/sub-01/T1maps/sub-01_ses-02_T1map_hybrid.nii.gz",
        "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-01_concentration.nii.gz",
        "mri-processed/mri_processed_data/sub-01/concentrations/sub-01_ses-02_concentration.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-intracranial_binary.nii.gz",
        "mri-processed/mri_dataset/derivatives/sub-01/ses-01/sub-01_ses-01_acq-mixed_T1map.nii.gz",
        "mri-processed/mri_dataset/derivatives/sub-01/ses-01/sub-01_ses-01_acq-looklocker_T1map.nii.gz",
        "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_acq-looklocker_T1map_registered.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf-aseg.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf-aparc+aseg.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-csf-wmparc.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aseg_refined.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-aparc+aseg_refined.nii.gz",
        "mri-processed/mri_processed_data/sub-01/segmentations/sub-01_seg-wmparc_refined.nii.gz",
        "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_T2w_registered.nii.gz",
        "mri-processed/mri_processed_data/sub-01/registered/sub-01_ses-01_T1w_registered.nii.gz",
        "freesurfer/mri_processed_data/freesurfer/sub-01/mri/aparc+aseg.mgz",
        "freesurfer/mri_processed_data/freesurfer/sub-01/mri/aseg.mgz",
        "freesurfer/mri_processed_data/freesurfer/sub-01/mri/wmparc.mgz",
    ]

    for file in files:
        src = inputdir / file
        dst = outdir / file

        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {src} to {dst}")
        dst.write_bytes(src.read_bytes())

    # Zip outdir into mritk-test-data.zip
    print("Creating zip archive...")
    zip_path = outdir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in outdir.rglob("*"):
            zipf.write(file, file.relative_to(outdir))


if __name__ == "__main__":
    main()
