"""Model loading and inference. Independent of Streamlit so it can be tested alone."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(weights_path: Path) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Train the model first (notebooks/02_model_training.ipynb) so that "
            "models/best.pt exists."
        )
    return YOLO(str(weights_path))


def run_detection(model: YOLO, image: Image.Image, conf: float = 0.25,
                  iou: float = 0.7) -> tuple[np.ndarray, list[dict]]:
    """Run YOLO on a PIL image.

    Returns (annotated RGB array, list of detections). Each detection is
    {"class": str, "confidence": float, "box": [x1, y1, x2, y2]} in pixels.
    """
    result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    annotated_bgr = result.plot()  # Ultralytics draws boxes + labels (BGR)
    annotated_rgb = annotated_bgr[:, :, ::-1]

    detections = []
    for box in result.boxes:
        detections.append({
            "class": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return np.ascontiguousarray(annotated_rgb), detections
