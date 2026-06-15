"""Smoke training: YOLO11n on coco128, 3 epochs, with W&B logging.

Purpose
-------
- Verify W&B integration works.
- Confirm training speed on the local accelerator.
- Produce the first metric chart in our W&B project.

Usage (from the app/ directory):
    uv run python scripts/train_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import wandb
from ultralytics import YOLO


PROJECT = "hometwin"          # W&B project name
RUN_NAME = "exp00_smoke_coco128"
# app/ root (this script lives in app/scripts/).
APP_ROOT = Path(__file__).resolve().parent.parent
# Single, consistent output location for every training run in this repo.
TRAIN_PROJECT = APP_ROOT / "runs" / "detect" / "runs" / "train"


def _default_device() -> str:
    """Pick CUDA if available, else Apple MPS, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    device = _default_device()
    # 1. Start a W&B run BEFORE Ultralytics so it auto-attaches.
    wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        config={
            "dataset": "coco128",
            "model": "yolo11n.pt",
            "epochs": 3,
            "imgsz": 320,
            "batch": 8,
            "device": device,
            "purpose": "pipeline smoke test",
        },
        tags=["smoke", "week1"],
    )

    # 2. Load pretrained YOLO11n and train.
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="coco128.yaml",   # ultralytics ships this; auto-downloads on first run
        epochs=3,
        imgsz=320,
        batch=8,
        device=device,
        project=str(TRAIN_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    print("---- training done ----")
    print(f"results saved at: {results.save_dir}")

    # 3. Log final mAP to W&B summary for easy comparison later.
    metrics = results.results_dict  # includes 'metrics/mAP50(B)' etc.
    for k, v in metrics.items():
        wandb.run.summary[k] = v
    wandb.finish()


if __name__ == "__main__":
    main()
