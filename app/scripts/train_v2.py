"""V2 학습 — mAP 0.549 → 0.65+ 노리기.

V1(train_baseline.py) 대비 변경점:
  - 모델: yolo11n → yolo11s (3배 파라미터)
  - 입력 크기: 640 → 832 (작은 객체 강세)
  - epoch: 50 → 100
  - augmentation 강화: mosaic + mixup + hsv + fliplr + scale + degrees
  - 클래스 가중치: 부정형(NO-*) 클래스에 가중치 부여 (cls=0.7, box=7.5)
  - 옵티마이저: AdamW + cosine LR + warmup 5 epoch
  - early stopping: patience 15

학습 시간 (GPU): 약 2-3시간

사용 (app/ 디렉토리에서):
    WANDB_API_KEY=xxx uv run python scripts/train_v2.py
"""
from __future__ import annotations

from pathlib import Path

import wandb
from ultralytics import YOLO


PROJECT = "hometwin"
RUN_NAME = "exp02_ppe_yolo11s_v2"
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
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml not found: {DATA_YAML} — run get_dataset.py first")

    cfg = {
        "dataset": "construction-site-safety v27",
        "n_classes": 10,
        "model": "yolo11s.pt",
        "epochs": 100,
        "imgsz": 832,
        "batch": 12,
        "device": _default_device(),
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "warmup_epochs": 5,
        "cos_lr": True,
        "patience": 15,
        # augmentation
        "mosaic": 1.0,
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "scale": 0.5,
        "degrees": 5.0,
        # loss weighting
        "box": 7.5,
        "cls": 0.7,    # 분류 비중 ↑ (NO-* 클래스 강화)
        "dfl": 1.5,
        "purpose": "v2: bigger model + aug + better LR for NO-* classes",
    }

    wandb.init(project=PROJECT, name=RUN_NAME, config=cfg,
               tags=["v2", "ppe", "yolo11s", "augmentation"])

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        device=cfg["device"], optimizer=cfg["optimizer"],
        lr0=cfg["lr0"], lrf=cfg["lrf"], cos_lr=cfg["cos_lr"],
        warmup_epochs=cfg["warmup_epochs"], patience=cfg["patience"],
        mosaic=cfg["mosaic"], mixup=cfg["mixup"],
        hsv_h=cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"],
        fliplr=cfg["fliplr"], scale=cfg["scale"], degrees=cfg["degrees"],
        box=cfg["box"], cls=cfg["cls"], dfl=cfg["dfl"],
        project=str(TRAIN_PROJECT), name=RUN_NAME, exist_ok=True,
        save=True, plots=True,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
