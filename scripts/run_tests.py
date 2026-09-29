"""Run the five qualitative tests and record what the model ACTUALLY outputs.

Put your test images in tests/images/ using these file-name prefixes:
    test1_clear*         clear image with common (trained-class) objects
    test2_multiple*      several objects in one image
    test3_small*         small / distant objects
    test4_unseen*        an object OUTSIDE the trained classes (e.g. a car, a dog)
    test5_lowquality*    blurry / dark / partially hidden

Optional helper to auto-make test5 from test1:
    python scripts/run_tests.py --degrade

Usage:   python scripts/run_tests.py [--conf 0.25]
Output:  outputs/metrics/test_results.md  (+ annotated images in outputs/predictions/tests/)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from PIL import Image, ImageFilter  # noqa: E402

from common import BEST_MODEL, IMG_EXTS, OUTPUTS_DIR, ROOT  # noqa: E402

TEST_DIR = ROOT / "tests" / "images"
CASES = [
    ("test1_clear", "Clear image, common objects"),
    ("test2_multiple", "Multiple objects"),
    ("test3_small", "Small / distant objects"),
    ("test4_unseen", "Object outside trained classes"),
    ("test5_lowquality", "Low quality / partially obscured"),
]


def degrade(src: Path, dst: Path) -> None:
    """Make a low-quality copy: downscale, blur, darken, black occluding box."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im = im.resize((max(w // 4, 64), max(h // 4, 64))).filter(ImageFilter.GaussianBlur(1.5))
    im = im.point(lambda p: int(p * 0.6))
    w, h = im.size
    for x in range(w // 3, 2 * w // 3):      # occlude the centre
        for y in range(h // 3, 2 * h // 3):
            im.putpixel((x, y), (0, 0, 0))
    im.save(dst)


def find_case_images(prefix: str) -> list[Path]:
    return sorted(p for p in TEST_DIR.glob(f"{prefix}*") if p.suffix.lower() in IMG_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.degrade:
        src = find_case_images("test1_clear")
        if not src:
            sys.exit("Need tests/images/test1_clear.* before using --degrade")
        degrade(src[0], TEST_DIR / "test5_lowquality_auto.png")

    from inference import load_model, run_detection
    model = load_model(BEST_MODEL)
    out_dir = OUTPUTS_DIR / "predictions" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Qualitative test results", "",
             f"Model: `{BEST_MODEL.name}`, confidence threshold: {args.conf}",
             "Observations are to be written by you after looking at the output images.", ""]
    for prefix, title in CASES:
        lines.append(f"## {title} (`{prefix}*`)")
        imgs = find_case_images(prefix)
        if not imgs:
            lines += [f"_No image found in tests/images/ with prefix `{prefix}`._", ""]
            continue
        for path in imgs:
            annotated, dets = run_detection(model, Image.open(path).convert("RGB"), conf=args.conf)
            Image.fromarray(annotated).save(out_dir / f"{path.stem}_pred.png")
            lines.append(f"- **Input:** `{path.name}` -> output `{path.stem}_pred.png`")
            if dets:
                for d in dets:
                    lines.append(f"  - {d['class']}: {d['confidence'] * 100:.1f}%")
            else:
                lines.append("  - (no detections)")
            lines.append("- **Observations:** _TODO: fill in after viewing the image_")
        lines.append("")

    report = OUTPUTS_DIR / "metrics" / "test_results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
