"""Small helpers for the Streamlit app."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Override with:  YOLO_WEIGHTS=/path/to/other.pt streamlit run app/app.py
MODEL_PATH = Path(os.environ.get("YOLO_WEIGHTS", PROJECT_ROOT / "models" / "best.pt"))


def load_image(file) -> Image.Image:
    """Open an uploaded file as RGB, respecting EXIF orientation (phone photos)."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def detections_to_df(detections: list[dict]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["Class", "Confidence"])
    return pd.DataFrame({
        "Class": [d["class"] for d in detections],
        "Confidence": [f"{d['confidence'] * 100:.1f}%" for d in detections],
    })
