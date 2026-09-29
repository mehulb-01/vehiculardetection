# Real-Time Vehicular & Road Scene Object Detection using YOLO11

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision system for real-time vehicular and road actor detection in urban environments. The system fine-tunes **Ultralytics YOLO11 (Nano)** on the **KITTI Road / Vehicle Benchmark** across 8 distinct classes and provides an interactive **Streamlit web application** with live threshold adjustment, webcam snapshot capabilities, and automated qualitative stress testing.

---

## Key Features

- **8 Road Scene Classes**: Detects and localizes `car`, `van`, `truck`, `pedestrian`, `Person_sitting`, `cyclist`, `tram`, and `misc`.
- **Pretrained Weights Included**: Ships with fine-tuned `models/best.pt` (~21 MB) ready for instant local or cloud deployment.
- **Automated Data Engineering**: Frictionless download and extraction of the KITTI benchmark directly via public release assets (no API keys or Kaggle credentials required).
- **Data Validation & Hygiene Pipeline**: Automated verification of image corruptions, label bounds checking $[0, 1]$, and deterministic 70/20/10 split generation (`seed=42`).
- **Interactive Streamlit Web Interface**:
  - Live confidence score slider ($0.05$ to $0.95$).
  - Side-by-side comparison of original and model-annotated bounding boxes.
  - Tabular breakdown of detected objects and confidence percentages.
  - Dual input modalities: Image file upload and browser-based webcam snapshots.
- **Rigorous Evaluation Suite**: Precision-Recall curves, confusion matrices, and 5 automated qualitative edge-case stress tests.

---

## Performance Metrics

Trained on an NVIDIA GPU using input resolution $640 \times 640$ pixels with Automatic Mixed Precision (`amp=True`):

| Evaluation Split | Precision ($P$) | Recall ($R$) | mAP@50 | mAP@50–95 |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Set** (1,496 images) | **68.3%** | **56.3%** | **61.5%** | **37.4%** |
| **Test Set** (749 images) | **67.0%** | **54.3%** | **60.2%** | **36.7%** |

> *The negligible generalization gap (<1.4% mAP@50 between validation and unseen test sets) confirms robust model stability without overfitting.*

---

## Project Structure

```text
vehiculardetection/
├── app/
│   ├── app.py                     # Streamlit web application frontend
│   ├── inference.py               # Cached model loader and inference runner
│   └── utils.py                   # Image loaders, EXIF orienters, DataFrame helpers
├── data/
│   ├── raw/                       # Downloaded archive (git-ignored)
│   └── dataset/                   # Cleaned YOLO format dataset & dataset.yaml
├── models/
│   └── best.pt                    # Fine-tuned YOLO11n weights checkpoint
├── notebooks/
│   ├── 01_dataset_analysis.ipynb  # Data download, integrity check & EDA
│   ├── 02_model_training.ipynb    # Transfer learning pipeline & checkpointing
│   └── 03_model_evaluation.ipynb  # Quantitative metrics, PR curves & confusion matrix
├── outputs/
│   ├── metrics/                   # metrics.json, confusion matrices, PR/F1 curves
│   ├── plots/                     # Dataset distribution & training history plots
│   └── predictions/               # Test image inference visualizations
├── scripts/
│   ├── check_env.py               # Hardware, CUDA, and package diagnostics
│   ├── common.py                  # Shared path utilities and dataset builders
│   ├── download_dataset.py        # Automated KITTI asset fetcher
│   └── run_tests.py               # Automated 5-scenario qualitative stress tester
├── tests/
│   └── images/                    # Qualitative test benchmarks (clear, dense, degraded)
├── requirements.txt               # Pinned project dependencies
└── README.md
```

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mehulb-01/vehiculardetection.git
cd vehiculardetection
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

---

## Running the Web Application

Launch the Streamlit interactive dashboard directly using the pre-packaged `models/best.pt`:

```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser to upload road photos or use your webcam.

To specify a custom weights file:
```bash
# Windows PowerShell
$env:YOLO_WEIGHTS="path/to/custom_weights.pt"; streamlit run app/app.py

# Linux / macOS
YOLO_WEIGHTS="path/to/custom_weights.pt" streamlit run app/app.py
```

---

## Reproducing the Pipeline (Training & Evaluation)

### Step 1: Download & Prepare the Dataset
```bash
python scripts/download_dataset.py
```
This fetches the KITTI dataset archive and extracts it to `data/raw/kitti/`.

### Step 2: Run Notebooks Sequentially
You can execute the notebooks interactively via Jupyter or headlessly via `nbconvert`:
```bash
# 1. Dataset Analysis, Integrity Verification, and 70/20/10 Split
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_analysis.ipynb

# 2. Transfer Learning with YOLO11 Nano (saves checkpoint to models/best.pt)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_model_training.ipynb

# 3. Comprehensive Model Evaluation & Metric Generation
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_evaluation.ipynb
```

---

## Qualitative Edge-Case Stress Testing

Run the stress-test suite across 5 realistic road scenarios:
1. **Clear Image**: Daylight conditions with standard road participants.
2. **Dense Scene**: Multiple vehicles with mutual occlusion.
3. **Small / Distant Objects**: Distant pedestrians and vehicles at scale.
4. **Out-of-Distribution (OOD)**: Subjects outside the 8 trained classes (tests false-positive rejection).
5. **Degraded Sensor**: Artificially blurred, darkened, downscaled, and occluded images (`--degrade`).

```bash
python scripts/run_tests.py --conf 0.25
```
Results and qualitative observations are logged to `outputs/metrics/test_results.md` and annotated visual outputs are saved to `outputs/predictions/tests/`.

---

## Technical Architecture

- **Backbone**: Modified CSPDarknet / C3k2 cross-stage modules for robust multi-scale spatial feature extraction.
- **Neck**: Path Aggregation Network (PANet) fusing low-level spatial geometry with high-level semantics.
- **Head**: Decoupled anchor-free detection head using Task-Aligned Assigner (TAL).
- **Loss Functions**:
  - Box Regression: Complete Intersection over Union ($CIoU$) + Distribution Focal Loss ($DFL$).
  - Classification: Binary Cross-Entropy ($BCE$).

---

## References & Acknowledgments

- **KITTI Vision Benchmark Suite**: [Geiger et al., *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012](https://www.cvlibs.net/datasets/kitti/).
- **Ultralytics YOLO11**: [Ultralytics GitHub Repository & Documentation](https://github.com/ultralytics/ultralytics).

---

## License

This project is licensed under the [MIT License](LICENSE).
