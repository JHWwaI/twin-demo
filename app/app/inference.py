"""PPE 검출 + 알람 로직 통합 추론 엔진.

학습된 best.pt 로 이미지/영상에서 안전장구를 검출하고,
"NO-Hardhat", "NO-Safety Vest" 가 감지되면 위험 카운트를 올린다.

타 모듈에서 import 해서 쓰는 공용 엔진.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


def get_default_device() -> str:
    """Return the best available inference device for the current machine.

    Preference order: CUDA GPU ("0") → Apple MPS → CPU. Importing torch lazily
    keeps this safe even if the accelerator backends are unavailable. This is the
    single source of truth for device selection across the app and scripts.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


CLASS_KOR = {
    "Hardhat": "헬멧",
    "Mask": "마스크",
    "NO-Hardhat": "헬멧 미착용",
    "NO-Mask": "마스크 미착용",
    "NO-Safety Vest": "조끼 미착용",
    "Person": "사람",
    "Safety Cone": "안전 콘",
    "Safety Vest": "안전 조끼",
    "machinery": "중장비",
    "vehicle": "차량",
}

# 위험으로 분류할 클래스 (이름 그대로)
HAZARD_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# --- BGR 색상 팔레트 (OpenCV 는 BGR 순서) ---
SAFE_BGR = (29, 158, 117)        # 초록  — 정상 착용 (헬멧/마스크/조끼)
ACCENT_BGR = (61, 163, 232)      # 주황  — 안전 콘
NEUTRAL_BGR = (188, 141, 90)     # 블루그레이 — 사람/장비/차량
HAZARD_BGR = (48, 90, 216)       # 빨강  — 위험(미착용) + 경보 바
DEFAULT_BGR = (200, 200, 200)    # 회색  — 알 수 없는 클래스 fallback
LABEL_TEXT_BGR = (255, 255, 255)  # 흰색  — 라벨/경보 글자

COLORS = {
    "Hardhat":         SAFE_BGR,
    "Mask":            SAFE_BGR,
    "Safety Vest":     SAFE_BGR,
    "Safety Cone":     ACCENT_BGR,
    "Person":          NEUTRAL_BGR,
    "machinery":       NEUTRAL_BGR,
    "vehicle":         NEUTRAL_BGR,
    "NO-Hardhat":      HAZARD_BGR,
    "NO-Mask":         HAZARD_BGR,
    "NO-Safety Vest":  HAZARD_BGR,
}

# --- 박스/라벨 렌더링 상수 ---
BOX_THICKNESS = 2                 # 일반 박스 선 두께
HAZARD_BOX_THICKNESS = 3          # 위험 박스는 더 두껍게 강조
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1

# --- 경보 오버레이 상수 ---
ALARM_BAR_HEIGHT = 56             # 상단 경보 바 높이(px)
ALARM_OVERLAY_ALPHA = 0.78        # 경보 바 불투명도 (1 - 0.22 배경)
ALARM_BG_ALPHA = 0.22             # 경보 바 아래 비치는 원본 비율
ALARM_FONT_SCALE = 1.0
ALARM_FONT_THICKNESS = 2


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def is_hazard(self) -> bool:
        return self.cls_name in HAZARD_CLASSES


@dataclass
class FrameResult:
    frame_idx: int
    detections: list[Detection] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def hazard_count(self) -> int:
        return sum(1 for d in self.detections if d.is_hazard)

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.cls_name == "Person")

    def class_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for d in self.detections:
            c[d.cls_name] = c.get(d.cls_name, 0) + 1
        return c


class PPEDetector:
    """YOLO11 기반 PPE 검출/추적 엔진.

    학습된 가중치를 한 번 로드해두고 이미지 단건 추론(:meth:`predict_image`)과
    비디오 스트리밍 추론(:meth:`stream_video`, ByteTrack 옵션)을 제공한다.
    원시 Ultralytics 결과를 :class:`FrameResult`(검출 + 위험 카운트)로 정규화해
    Streamlit 등 상위 모듈이 동일한 인터페이스로 소비하도록 한다.
    """

    def __init__(self, weights: str | Path, device: str | None = None):
        """가중치를 로드한다.

        Args:
            weights: 학습된 .pt 파일 경로.
            device: 추론 디바이스. None 이면 :func:`get_default_device` 로
                CUDA→MPS→CPU 순으로 자동 선택한다.
        """
        self.model = YOLO(str(weights))
        self.device = device or get_default_device()
        self.names = self.model.names

    def predict_image(
        self, img: np.ndarray, conf: float = 0.25, imgsz: int = 640
    ) -> FrameResult:
        """단일 BGR 이미지를 추론해 :class:`FrameResult` 로 반환한다."""
        t0 = time.time()
        r = self.model.predict(
            img, conf=conf, imgsz=imgsz, device=self.device, verbose=False
        )[0]
        dets = self._parse(r)
        return FrameResult(
            frame_idx=0, detections=dets, elapsed_ms=(time.time() - t0) * 1000
        )

    def stream_video(
        self,
        src: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        track: bool = True,
        vid_stride: int = 1,
    ) -> Iterable[tuple[np.ndarray, FrameResult]]:
        """비디오를 스트리밍 추론해 프레임별 ``(frame_bgr, FrameResult)`` 를 yield.

        Args:
            src: 비디오 파일 경로.
            conf: confidence 임계값.
            imgsz: 추론 입력 크기.
            track: True 면 ByteTrack 으로 객체에 일관된 track_id 부여,
                False 면 추적 없이 프레임 단위 검출만 수행.
            vid_stride: 프레임 샘플링 간격(1 = 모든 프레임).
        """
        if track:
            stream = self.model.track(
                source=str(src), conf=conf, imgsz=imgsz, device=self.device,
                tracker="bytetrack.yaml", stream=True, verbose=False,
                vid_stride=vid_stride,
            )
        else:
            stream = self.model.predict(
                source=str(src), conf=conf, imgsz=imgsz, device=self.device,
                stream=True, verbose=False, vid_stride=vid_stride,
            )
        for i, r in enumerate(stream):
            yield r.orig_img, FrameResult(
                frame_idx=i, detections=self._parse(r),
                elapsed_ms=float(r.speed.get("inference", 0.0)),
            )

    def _parse(self, r) -> list[Detection]:
        """Ultralytics Results 한 개를 :class:`Detection` 리스트로 변환한다."""
        dets: list[Detection] = []
        if r.boxes is None:
            return dets
        for i, b in enumerate(r.boxes):
            cls_id = int(b.cls.item())
            name = self.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            tid = None
            if b.id is not None:
                tid = int(b.id.item())
            dets.append(Detection(
                cls_name=name, conf=float(b.conf.item()),
                x1=x1, y1=y1, x2=x2, y2=y2, track_id=tid,
            ))
        return dets


# -------- 시각화 --------

def draw_boxes(img: np.ndarray, result: FrameResult, show_alarm: bool = True) -> np.ndarray:
    """검출 박스 + 라벨(+track_id)을 그린 새 이미지를 반환한다.

    위험 클래스는 더 두꺼운 박스로 강조하고, ``show_alarm`` 이 True 이며 위험
    검출이 하나라도 있으면 상단에 경보 바를 합성한다. 원본은 변경하지 않는다.
    """
    out = img.copy()
    for d in result.detections:
        color = COLORS.get(d.cls_name, DEFAULT_BGR)
        thick = HAZARD_BOX_THICKNESS if d.is_hazard else BOX_THICKNESS
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      color, thick)
        label = f"{d.cls_name} {d.conf:.2f}"
        if d.track_id is not None:
            label = f"#{d.track_id} " + label
        # 라벨 배경
        (tw, th), _ = cv2.getTextSize(
            label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
        cv2.rectangle(out, (int(d.x1), int(d.y1) - th - 6),
                      (int(d.x1) + tw + 4, int(d.y1)), color, -1)
        cv2.putText(out, label, (int(d.x1) + 2, int(d.y1) - 4),
                    LABEL_FONT, LABEL_FONT_SCALE, LABEL_TEXT_BGR,
                    LABEL_FONT_THICKNESS, cv2.LINE_AA)

    if show_alarm and result.hazard_count > 0:
        _overlay_alarm(out, result.hazard_count)
    return out


def _overlay_alarm(img: np.ndarray, n: int) -> None:
    """상단에 반투명 빨간 경보 바와 위반 건수 텍스트를 in-place 로 합성한다."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, ALARM_BAR_HEIGHT), HAZARD_BGR, -1)
    cv2.addWeighted(overlay, ALARM_OVERLAY_ALPHA, img, ALARM_BG_ALPHA, 0, dst=img)
    text = f"WARNING  PPE non-compliance  {n}"
    cv2.putText(img, text, (16, 38), LABEL_FONT,
                ALARM_FONT_SCALE, LABEL_TEXT_BGR, ALARM_FONT_THICKNESS, cv2.LINE_AA)
