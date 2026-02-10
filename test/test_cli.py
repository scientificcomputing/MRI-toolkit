import json
import mritk
import mritk.cli as cli


def test_cli_version(capsys):
    cli.main(["--version"])
    captured = capsys.readouterr()
    assert "MRI Toolkit Environment" in captured.out
    assert f"mri-toolkit │   {mritk.__version__} │" in captured.out


def test_cli_info(capsys, mri_data_dir):
    test_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    args = ["info", str(test_file)]
    cli.main(args)
    captured = capsys.readouterr()
    assert "Voxel Size (mm)   (0.50, 0.50, 0.50)" in captured.out
    assert "Shape (x, y, z)   (368, 512, 512)" in captured.out


def test_cli_info_json(capsys, mri_data_dir):
    test_file = mri_data_dir / "mri-processed/mri_processed_data/sub-01" / "concentrations/sub-01_ses-01_concentration.nii.gz"
    args = ["info", str(test_file), "--json"]
    cli.main(args)
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert "sub-01_ses-01_concentration.nii.gz" in data["filename"]
    assert data["shape"] == [368, 512, 512]
