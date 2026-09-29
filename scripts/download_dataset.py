"""Download and extract the KITTI Road / Vehicle dataset (Ultralytics-hosted).

No Kaggle account or API key is needed: the archive is a public GitHub
release asset. Usage:   python scripts/download_dataset.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip"
ZIP_PATH = RAW_DIR / "kitti.zip"
EXTRACT_DIR = RAW_DIR / "kitti"


def download(url: str = URL, zip_path: Path = ZIP_PATH) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        print(f"Already downloaded: {zip_path}")
        return zip_path
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:  # network / DNS / proxy problems
        zip_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed: {e}\nCheck your internet connection, or download the "
            f"file manually from the URL above and save it as {zip_path}"
        )
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise SystemExit("Downloaded file is not a valid zip archive.")
    return zip_path


def extract(zip_path: Path = ZIP_PATH, out_dir: Path = EXTRACT_DIR) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Already extracted: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")
    return out_dir


if __name__ == "__main__":
    extract(download())
