"""Run YOLO on a video and save a copy with boxes drawn on detected objects.

Usage (from the app/ directory):
    uv run python scripts/draw_boxes.py footage/warehouse_01.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

# app/app holds the shared inference module (get_default_device lives there).
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT / "app"))
from inference import get_default_device  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: uv run python scripts/draw_boxes.py <video_path>")
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        sys.exit(f"[error] file not found: {src}")

    # YOLO11n — smallest, fastest. Pretrained on COCO (80 classes incl. person, truck, etc.)
    model = YOLO("yolo11n.pt")

    # CUDA if available, else Apple MPS, else CPU — chosen at runtime.
    device = get_default_device()
    print(f"[1/2] running YOLO11n on {src.name} (device={device}) ...")
    results = model.predict(
        source=str(src),
        device=device,
        save=True,             # writes annotated video to runs/detect/predict/
        project=str(src.parent.parent / "outputs"),
        name="boxes",
        exist_ok=True,
        conf=0.35,
        vid_stride=2,          # process every 2nd frame -> ~2x faster, still smooth
    )

    out_dir = Path(results[0].save_dir)
    out_files = sorted(out_dir.glob("*.mp4")) + sorted(out_dir.glob("*.avi"))
    print("[2/2] done. annotated file(s):")
    for f in out_files:
        print(f"  -> {f}")


if __name__ == "__main__":
    main()
