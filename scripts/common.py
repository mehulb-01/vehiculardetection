"""Shared helpers for the YOLO object-detection project.

Everything is pathlib-based so it works on Windows, Linux and macOS.
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = ROOT / "data" / "dataset"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BEST_MODEL = MODELS_DIR / "best.pt"

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fallback class names for KITTI road/vehicle dataset:
# car, van, truck, pedestrian, person_sitting, cyclist, tram, misc
FALLBACK_CLASS_NAMES = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]

SPLIT_ALIASES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
    "test": ["test"],
}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def find_splits(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    """Locate (images_dir, labels_dir) per split inside a raw dataset folder.

    Supports both layouts:  <split>/images + <split>/labels
                       and  images/<split> + labels/<split>
    """
    found: dict[str, tuple[Path, Path]] = {}
    dirs = [d for d in raw_root.rglob("*") if d.is_dir()]
    for split, aliases in SPLIT_ALIASES.items():
        for d in dirs:
            if d.name not in aliases:
                continue
            a_img, a_lbl = d / "images", d / "labels"  # layout A
            if a_img.is_dir() and a_lbl.is_dir():
                found[split] = (a_img, a_lbl)
                break
            parent = d.parent  # layout B: .../images/<split>
            if parent.name == "images" and (parent.parent / "labels" / d.name).is_dir():
                found[split] = (d, parent.parent / "labels" / d.name)
                break
    return found


def validate_label_file(path: Path, num_classes: int) -> tuple[list[str], list[str]]:
    """Return (valid_lines, problems) for one YOLO label file."""
    good, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{path.name}:{i} expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            problems.append(f"{path.name}:{i} non-numeric value")
            continue
        if not (0 <= cls < num_classes):
            problems.append(f"{path.name}:{i} class id {cls} out of range")
        elif not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            problems.append(f"{path.name}:{i} box values outside (0,1]")
        else:
            good.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return good, problems


def _image_ok(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_yolo_dataset(raw_root: Path, out_dir: Path, class_names: list[str],
                       seed: int = SEED) -> dict:
    """Validate the raw dataset and write a clean YOLO structure + dataset.yaml.

    - Keeps existing train/val/test splits when all three exist.
    - Otherwise pools everything and makes a reproducible 70/20/10 split.
    - Drops corrupt images, images without labels, malformed label lines.
    Returns a report dict with counts and problems.
    """
    num_classes = len(class_names)
    splits = find_splits(raw_root)
    report = {"preserved_splits": set(splits) >= {"train", "val", "test"},
              "problems": [], "dropped_images": [], "counts": {}}

    # Collect (split -> list of (image, label)) pairs.
    pairs: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    if report["preserved_splits"]:
        for split, (img_dir, lbl_dir) in splits.items():
            for img in list_images(img_dir):
                pairs[split].append((img, lbl_dir / f"{img.stem}.txt"))
    else:
        pool = []
        for img_dir, lbl_dir in splits.values():
            pool += [(i, lbl_dir / f"{i.stem}.txt") for i in list_images(img_dir)]
        if not pool:
            raise FileNotFoundError(f"No images/labels found under {raw_root}")
        random.Random(seed).shuffle(pool)
        n = len(pool)
        a, b = int(0.7 * n), int(0.9 * n)
        pairs["train"], pairs["val"], pairs["test"] = pool[:a], pool[a:b], pool[b:]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, items in pairs.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        kept = 0
        for img, lbl in items:
            if not _image_ok(img):
                report["dropped_images"].append(f"{img.name}: corrupt image")
                continue
            if not lbl.exists():
                report["dropped_images"].append(f"{img.name}: missing label file")
                continue
            lines, probs = validate_label_file(lbl, num_classes)
            report["problems"] += probs
            if not lines:
                report["dropped_images"].append(f"{img.name}: no valid boxes")
                continue
            shutil.copy2(img, out_dir / "images" / split / img.name)
            (out_dir / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
        report["counts"][split] = kept

    write_dataset_yaml(out_dir, class_names)
    return report


def write_dataset_yaml(out_dir: Path, class_names: list[str]) -> Path:
    lines = [f"path: {out_dir.resolve().as_posix()}", "train: images/train",
             "val: images/val", "test: images/test", "", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(class_names)]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def class_distribution(dataset_dir: Path, class_names: list[str]) -> dict[str, Counter]:
    """Boxes per class for each split."""
    dist = {}
    for split in ("train", "val", "test"):
        c: Counter = Counter()
        for f in (dataset_dir / "labels" / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    c[class_names[int(line.split()[0])]] += 1
        dist[split] = c
    return dist
