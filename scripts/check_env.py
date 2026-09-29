"""Verify that the environment is ready:  python scripts/check_env.py"""
import importlib
import sys

REQUIRED = ["torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas",
            "matplotlib", "PIL", "streamlit"]

missing = []
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        print(f"OK   {name:12s} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"MISSING {name}")
        missing.append(name)

if missing:
    sys.exit("Install missing packages with: pip install -r requirements.txt")

import torch  # noqa: E402

print("Python:", sys.version.split()[0])
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0), f"({props.total_memory / 1e9:.1f} GB)")
else:
    print("GPU: CPU only -> training will use reduced settings (see notebook 02)")

from ultralytics import YOLO  # noqa: E402
print("Ultralytics import OK:", YOLO.__name__)
