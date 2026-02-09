"""MRI -- Download data for testing

Copyright (C) 2026   Henrik Finsberg (henriknf@simula.no)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor
import tqdm

logger = logging.getLogger(__name__)


# def download_data(outdir: Path, file_info: tuple) -> None:
def download_data(args) -> None:
    (outdir, file_info) = args
    (filename, url) = file_info
    output_path = outdir / Path(filename).stem / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} to {output_path}. This may take a while.")

    try:
        urlretrieve(url, output_path)
        if not zipfile.is_zipfile(output_path):
            logger.info(f"Downloaded {filename} is not a zip file. No extraction needed.")
            return
        logger.info(f"Extracting {output_path} to {output_path.parent}.")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(output_path.parent)
        output_path.unlink()
    except Exception as e:
        logger.error(f"Failed to download {filename} from {url}. Error: {e}")
        return None
    return output_path


# Download multiple files concurrently
# Implementation inspired by https://medium.com/@ryan_forrester_/downloading-files-from-urls-in-python-f644e04a0b16
def download_multiple(urls: dict, outdir, max_workers=4):
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for thread pool
    args = [(outdir, file_info) for file_info in urls.items()]

    # Download files using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(download_data, args),
                total=len(urls),
                desc="Downloading MRI data - Gonzo",
            )
        )

    # Process results
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)

    print("\nDownload complete:")
    print(f"- Successfully downloaded: {len(successful)} files")
    print(f"- Failed downloads: {failed} files")

    return successful
