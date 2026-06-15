"""Week 1 baseline: YOLO11n fine-tune on Construction PPE (10 classes).

This is exp01 — the first 'real' training run for the portfolio.
Logged to W&B project 'hometwin' as run 'exp01_ppe_yolo11n'.

Usage (from the app/ directory):
    WANDB_API_KEY=xxx uv run python scripts/train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import wandb
from ultralytics import YOLO


PROJECT = "hometwin"
RUN_NAME = "exp01_ppe_yolo11n"
# app/ root (this script lives in app/scripts/).
APP_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = APP_ROOT / "datasets" / "construction-ppe" / "data.yaml"
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
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml not found: {DATA_YAML} — run get_dataset.py first")

    wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        config={
            "dataset": "construction-site-safety v27",
            "n_classes": 10,
            "model": "yolo11n.pt",
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "device": device,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "purpose": "baseline finetune on PPE dataset",
        },
        tags=["baseline", "ppe", "week1", "yolo11n"],
    )

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        project=str(TRAIN_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        patience=15,            # early stop if no improvement for 15 epochs
        save_period=10,         # checkpoint every 10 epochs
    )

    print("---- training done ----")
    print(f"results saved at: {results.save_dir}")

    # Push final metrics to W&B summary for the leaderboard later.
    for k, v in results.results_dict.items():
        wandb.run.summary[k] = v
    wandb.finish()


if __name__ == "__main__":
    main()
