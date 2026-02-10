import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import your module
import mritk.cli
import mritk.datasets
from mritk.datasets import Dataset

# --- Fixtures ---


@pytest.fixture
def mock_datasets():
    """Returns a simplified version of the dataset dictionary using the new Dataclass."""
    return {
        "test-data": Dataset(
            name="Test Data",
            description="Test description",
            doi="10.1234/test",
            license="MIT",
            links={
                "file1.txt": "http://example.com/file1.txt",
                "archive.zip": "http://example.com/archive.zip",
            },
        ),
        "gonzo": Dataset(name="Gonzo", description="Gonzo description", links={}),
    }


def test_get_datasets_structure():
    """Ensure get_datasets returns a dict of Dataset objects."""
    datasets = mritk.datasets.get_datasets()
    assert "test-data" in datasets
    assert isinstance(datasets["test-data"], Dataset)
    assert datasets["test-data"].doi == "10.5281/zenodo.14266867"


def test_add_arguments():
    """Ensure argparse is configured correctly with subcommands."""
    import argparse

    parser = argparse.ArgumentParser()
    mritk.datasets.add_arguments(parser)

    # Test 'download' subcommand
    args = parser.parse_args(["download", "test-data", "-o", "/tmp/out"])
    # Note: argparse converts hyphens in 'dest' to underscores for attributes,
    # but strictly speaking, add_subparsers dest keeps the name if accessed via vars()
    # Let's check the logic used in the script.
    assert getattr(args, "datasets-command") == "download"
    assert args.dataset == "test-data"
    assert args.outdir == Path("/tmp/out")

    # Test 'list' subcommand
    args_list = parser.parse_args(["list"])
    assert getattr(args_list, "datasets-command") == "list"


@patch("mritk.datasets.get_datasets")
@patch("mritk.datasets.download_multiple")
def test_dispatch_download_success(mock_download_multiple, mock_get_datasets, mock_datasets):
    """Test that dispatch calls download_multiple with the .links attribute."""
    mock_get_datasets.return_value = mock_datasets

    mritk.cli.main(["datasets", "download", "test-data", "-o", "/tmp"])

    # CRITICAL: This test asserts that you passed the .links dictionary,
    # not the Dataset object itself.
    expected_links = mock_datasets["test-data"].links
    mock_download_multiple.assert_called_once_with(expected_links, Path("/tmp"))


@patch("mritk.datasets.list_datasets")
def test_dispatch_list(mock_list_datasets):
    """Test that dispatch calls list_datasets when subcommand is 'list'."""
    mritk.cli.main(["datasets", "list"])
    mock_list_datasets.assert_called_once()


@patch("mritk.datasets.get_datasets")
def test_dispatch_unknown_subcommand(mock_get_datasets):
    """Test graceful failure on unknown subcommand."""
    args = {"datasets-command": "unknown"}
    with pytest.raises(ValueError) as excinfo:
        mritk.datasets.dispatch(args)
    assert "Unknown subcommand" in str(excinfo.value)


@patch("rich.console.Console")  # Mock rich.console.Console
@patch("mritk.datasets.get_datasets")
def test_list_datasets(mock_get_datasets, mock_console_cls, mock_datasets):
    """Test that list_datasets attempts to print to console."""
    mock_get_datasets.return_value = mock_datasets
    mock_console_instance = mock_console_cls.return_value

    mritk.datasets.list_datasets()

    # Verify it tried to print something
    assert mock_console_instance.print.called
    # We can verify it printed the dataset name
    # (Checking exact rich output is hard, checking invocation is usually enough)
    assert mock_console_cls.called


@patch("urllib.request.urlretrieve")
@patch("zipfile.is_zipfile")
def test_download_data_regular_file(mock_is_zip, mock_retrieve, tmp_path):
    """Test downloading a standard non-zip file."""
    mock_is_zip.return_value = False
    filename = "test.txt"
    url = "http://example.com/test.txt"
    outdir = tmp_path / "output"

    args = (outdir, (filename, url))

    result_path = mritk.datasets.download_data(args)

    expected_path = outdir / "test" / filename
    assert result_path == expected_path
    mock_retrieve.assert_called_once()


@patch("urllib.request.urlretrieve")
def test_download_data_failure(mock_retrieve, tmp_path, caplog):
    """Test error handling - now expects an Exception to be raised."""
    mock_retrieve.side_effect = Exception("Connection Reset")

    args = (tmp_path, ("file.txt", "http://bad-url.com"))

    # The new code re-raises the exception, so we use pytest.raises
    with pytest.raises(Exception) as excinfo:
        mritk.datasets.download_data(args)

    assert "Connection Reset" in str(excinfo.value)
    # Also verify it logged the error before raising
    assert "Failed to download" in caplog.text


@patch("mritk.datasets.ThreadPoolExecutor")
@patch("mritk.datasets.download_data")
def test_download_multiple(mock_download_data, mock_executor, tmp_path):
    """Test the threading logic."""
    # Note: inputs are now a dict of links, not a Dataset object
    urls = {"f1": "u1", "f2": "u2"}

    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    mock_executor_instance.map.return_value = ["path/to/f1", "path/to/f2"]

    successful = mritk.datasets.download_multiple(urls, tmp_path)

    assert len(successful) == 2
