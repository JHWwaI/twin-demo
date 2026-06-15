"""PPE Vision Twin — Streamlit 통합 데모.

탭 4개:
  1. 이미지 분석   : 업로드 → 검출 + 알람 + 카운트
  2. 영상 분석     : 업로드 → 추적 + 알람 + 시계열
  3. 2D 평면 트윈  : 검출 좌표를 평면도에 매핑 (Plotly)
  4. 학습 결과     : confusion matrix / PR curve / val_pred 이미지

실행 (app/ 디렉토리에서):
    uv sync
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from inference import (  # noqa: E402
    CLASS_KOR, HAZARD_CLASSES,
    FrameResult, PPEDetector, draw_boxes, get_default_device,
)

# ---- 가중치 경로 ----
# 우선순위: 저장소에 포함된 깨끗한 V1 가중치 → (없으면) 학습 산출물 디렉토리.
# 학습 run 디렉토리는 .gitignore 대상이라 클론 직후엔 없을 수 있으므로 fallback.
WEIGHTS_CANDIDATES = [
    ROOT / "models" / "best_v1.pt",
    ROOT / "runs" / "detect" / "runs" / "train" / "exp01_ppe_yolo11n" / "weights" / "best.pt",
]
WEIGHTS = next((p for p in WEIGHTS_CANDIDATES if p.exists()), WEIGHTS_CANDIDATES[0])
EVAL_DIR = ROOT / "runs" / "detect" / "outputs" / "ppe_eval"
DOCS_METRICS_DIR = ROOT.parent / "docs" / "metrics"

# ---- UI 상수 ----
# Plotly 시계열 색상 (작업자 vs 위험).
COLOR_PERSON = "#5A8DBC"   # 블루그레이 — 작업자 라인
COLOR_HAZARD = "#D85A30"   # 주황빨강 — 위험 라인 / 위험 상태
COLOR_SAFE = "#1D9E75"     # 초록 — 정상 상태
COLOR_WARN = "#E8A33D"     # 노랑 — 중간(보통) 등급

# 업로드 영상 최대 크기 (MB). 초과 시 처리 거부.
MAX_VIDEO_MB = 200
# 차트 갱신 주기 (N 프레임마다 시계열 다시 그림).
CHART_REFRESH_EVERY = 5
# 발 위치 근사 시 마커 기본/최대 크기.
TWIN_MARKER_BASE = 10
TWIN_MARKER_RANGE = 30

st.set_page_config(
    page_title="PPE Vision Twin",
    page_icon="🦺",
    layout="wide",
)

# ---- 다크 톤 ----
st.markdown("""
<style>
.stApp { background-color: #1F2225; color: #E6E6E6; }
[data-testid="stSidebar"] { background-color: #25282C; }
.metric-card { background:#25282C; border:1px solid #3A3D42; border-radius:8px;
  padding:12px 16px; }
h1, h2, h3, h4 { color:#E6E6E6; }
.hazard-pill { background:#D85A30; color:white; padding:4px 10px;
  border-radius:12px; font-weight:600; }
.safe-pill { background:#1D9E75; color:white; padding:4px 10px;
  border-radius:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🦺 PPE Vision Twin")
st.caption("건설/공장 안전장구 검출 · 추적 · 위험 알람 · 2D 트윈 통합 데모")


# ---- 모델 로딩 (캐시) ----
@st.cache_resource
def load_model(weights: str, device: str) -> PPEDetector:
    """학습된 가중치로 :class:`PPEDetector` 를 생성한다 (Streamlit 캐시).

    weights/device 가 동일하면 재실행 시에도 모델을 다시 로드하지 않는다.
    """
    return PPEDetector(weights, device=device)


# ---------- 공용 렌더 헬퍼 ----------
def _render_metrics(res: FrameResult) -> None:
    """이미지 탭용 요약 메트릭 (검출/작업자/위험) 카드를 그린다."""
    cols = st.columns(3)
    cols[0].metric("검출 객체", len(res.detections))
    cols[1].metric("작업자(Person)", res.person_count)
    cols[2].metric("위험 신호", res.hazard_count,
                   delta="⚠️ 알람" if res.hazard_count else None,
                   delta_color="inverse")
    st.caption(f"추론 {res.elapsed_ms:.0f} ms")


def _render_class_table(res: FrameResult) -> None:
    """클래스별 검출 수 표를 그린다 (위험 클래스는 ⚠️ 표시)."""
    counts = res.class_counts()
    if not counts:
        st.info("검출된 객체가 없습니다.")
        return
    df = pd.DataFrame([
        {
            "클래스": k,
            "한글명": CLASS_KOR.get(k, k),
            "검출 수": v,
            "위험": "⚠️" if k in HAZARD_CLASSES else "",
        }
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)


# ---------- TAB 1: 이미지 ----------
def render_tab_image(detector: PPEDetector, conf: float, imgsz: int,
                     show_alarm: bool) -> None:
    """이미지 업로드 → 검출 + 알람 + 카운트 탭."""
    st.subheader("이미지 업로드 → 검출 + 알람")
    up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"], key="img_up")
    if up is None:
        return
    arr = np.frombuffer(up.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("이미지 로드 실패")
        return
    res = detector.predict_image(bgr, conf=conf, imgsz=imgsz)
    out = draw_boxes(bgr, res, show_alarm=show_alarm)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        _render_metrics(res)
        _render_class_table(res)


# ---------- TAB 2: 영상 + 추적 ----------
def render_tab_video(detector: PPEDetector, conf: float, imgsz: int,
                     show_alarm: bool) -> None:
    """영상 업로드 → ByteTrack 추적 + 시계열 위험 통계 탭."""
    st.subheader("영상 업로드 → ByteTrack 추적 + 시계열 위험 통계")
    up = st.file_uploader("MP4 / MOV", type=["mp4", "mov", "avi"], key="vid_up")
    use_track = st.checkbox("ByteTrack 추적 사용", value=True,
                            help="같은 사람에게 같은 ID 부여 → 시간 통계 가능")
    vid_stride = st.slider("프레임 간격 (1 = 모든 프레임)", 1, 10, 3)

    if up is None:
        return

    data = up.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        st.error(f"영상이 너무 큽니다 ({size_mb:.0f} MB). "
                 f"최대 {MAX_VIDEO_MB} MB 까지 지원합니다.")
        return

    video_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(data)
            video_path = f.name

        ph_video = st.empty()
        ph_metrics = st.empty()
        ph_chart = st.empty()

        timeline: list[dict] = []
        track_history: dict[int, dict] = {}  # track_id → {seen, hazard_seen, last_box}

        for bgr, res in detector.stream_video(
            video_path, conf=conf, imgsz=imgsz, track=use_track,
            vid_stride=vid_stride,
        ):
            out = draw_boxes(bgr, res, show_alarm=show_alarm)
            ph_video.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                           use_container_width=True)

            with ph_metrics.container():
                cols = st.columns(4)
                cols[0].metric("프레임", res.frame_idx)
                cols[1].metric("작업자", res.person_count)
                cols[2].metric("위험 신호", res.hazard_count,
                               delta="⚠️" if res.hazard_count else None,
                               delta_color="inverse")
                cols[3].metric("추론 ms", f"{res.elapsed_ms:.0f}")

            for d in res.detections:
                if d.track_id is None:
                    continue
                t = track_history.setdefault(d.track_id, {
                    "seen": 0, "hazard_seen": 0, "last_class": d.cls_name,
                    "last_box": (d.x1, d.y1, d.x2, d.y2),
                })
                t["seen"] += 1
                if d.is_hazard:
                    t["hazard_seen"] += 1
                t["last_class"] = d.cls_name
                t["last_box"] = (d.x1, d.y1, d.x2, d.y2)

            timeline.append({
                "frame": res.frame_idx,
                "person": res.person_count,
                "hazard": res.hazard_count,
                "total": len(res.detections),
            })
            if res.frame_idx % CHART_REFRESH_EVERY == 0 and timeline:
                df = pd.DataFrame(timeline)
                fig = px.line(df, x="frame", y=["person", "hazard"],
                              color_discrete_map={"person": COLOR_PERSON,
                                                  "hazard": COLOR_HAZARD},
                              title="프레임별 작업자/위험 카운트")
                fig.update_layout(template="plotly_dark", height=240,
                                  margin=dict(l=10, r=10, t=40, b=10))
                ph_chart.plotly_chart(fig, use_container_width=True)

        # 영상 종료 후 추적 요약
        if track_history:
            st.subheader("🎯 트랙별 요약 (ByteTrack)")
            df = pd.DataFrame([
                {
                    "ID": tid,
                    "마지막 클래스": h["last_class"],
                    "감지 프레임": h["seen"],
                    "위험 프레임": h["hazard_seen"],
                    "위험 비율": f"{(h['hazard_seen']/h['seen']*100):.0f}%",
                }
                for tid, h in track_history.items()
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.session_state["track_history"] = track_history
    finally:
        # 업로드 임시 파일 누수 방지.
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass


# ---------- TAB 3: 2D 평면 트윈 ----------
def render_tab_twin() -> None:
    """검출 좌표를 평면도에 매핑하는 2D 트윈 탭 (Tab 2 결과 사용)."""
    st.subheader("🗺️ 2D 평면 트윈 — 작업자 위치를 위에서 본 평면도에 매핑")
    st.caption("Tab 2 영상 분석을 먼저 실행하면 마지막 프레임의 위치가 표시됩니다.")
    th = st.session_state.get("track_history", {})
    if not th:
        st.info("Tab 2에서 영상을 먼저 분석해주세요.")
        return

    rows = []
    for tid, h in th.items():
        x1, y1, x2, y2 = h["last_box"]
        cx, cy = (x1 + x2) / 2, y2  # 발 위치 근사
        rows.append({
            "track_id": tid, "x": cx, "y": cy,
            "class": h["last_class"],
            "status": "위험" if h["last_class"] in HAZARD_CLASSES else "정상",
            "size": TWIN_MARKER_BASE
                    + (h["hazard_seen"] / max(h["seen"], 1)) * TWIN_MARKER_RANGE,
        })
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="x", y="y", color="status",
        size="size", text="track_id",
        color_discrete_map={"정상": COLOR_SAFE, "위험": COLOR_HAZARD},
        title="현장 평면도 — 작업자 마지막 위치 (Y 반전)",
    )
    fig.update_yaxes(autorange="reversed")  # 영상 좌표계 (y 아래로 증가)
    fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

    risky = df[df["status"] == "위험"]
    if not risky.empty:
        st.error(f"⚠️ 위험 작업자 {len(risky)}명: ID {', '.join(map(str, risky.track_id))}")
    else:
        st.success("✅ 위험 작업자 없음")


# ---------- TAB 4: 학습 결과 ----------
def render_tab_metrics() -> None:
    """학습된 모델의 검증 지표 + 산출물 이미지 탭."""
    st.subheader("학습된 모델 검증 지표")
    st.caption(f"가중치: `{WEIGHTS.name}`")

    metric_cols = st.columns(4)
    metric_cols[0].metric("전체 mAP@50", "0.549")
    metric_cols[1].metric("mAP@50-95", "0.273")
    metric_cols[2].metric("검증 이미지", "114")
    metric_cols[3].metric("Instance", "697")

    st.divider()
    st.markdown("### 클래스별 성능 (mAP@50)")
    per_cls = pd.DataFrame([
        ("Mask",           0.765, "잘함"),
        ("Safety Cone",    0.735, "잘함"),
        ("Hardhat",        0.686, "잘함"),
        ("machinery",      0.677, "잘함"),
        ("Person",         0.666, "잘함"),
        ("Safety Vest",    0.556, "보통"),
        ("NO-Mask",        0.424, "약함"),
        ("NO-Safety Vest", 0.355, "약함"),
        ("NO-Hardhat",     0.342, "약함"),
        ("vehicle",        0.284, "약함"),
    ], columns=["class", "mAP50", "tier"])
    fig = px.bar(per_cls, x="mAP50", y="class", orientation="h",
                 color="tier",
                 color_discrete_map={"잘함": COLOR_SAFE, "보통": COLOR_WARN,
                                     "약함": COLOR_HAZARD})
    fig.update_layout(template="plotly_dark", height=400,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # 검증 산출물 이미지: 학습 run 의 ppe_eval, 없으면 저장소의 docs/metrics 사용.
    img_dir = EVAL_DIR if EVAL_DIR.exists() else DOCS_METRICS_DIR
    if img_dir.exists():
        st.divider()
        st.markdown("### 검증 산출물")
        for name in ["confusion_matrix_normalized.png", "BoxPR_curve.png",
                     "BoxF1_curve.png", "val_batch0_pred.jpg",
                     "val_batch1_pred.jpg"]:
            p = img_dir / name
            if p.exists():
                with st.expander(name, expanded=False):
                    st.image(str(p), use_container_width=True)


def main() -> None:
    """사이드바 설정을 읽고 모델을 로드한 뒤 4개 탭을 렌더링한다."""
    if not WEIGHTS.exists():
        st.error(f"학습된 가중치가 없습니다: {WEIGHTS}\n"
                 "scripts/train_baseline.py 또는 scripts/train_v2.py 로 먼저 학습하세요.")
        st.stop()

    # ---- 사이드바 ----
    device_options = ["cuda", "cpu", "mps"]
    default_device = get_default_device()
    # get_default_device 는 cuda 를 "0" 으로 반환 → selectbox 표기는 "cuda".
    default_label = "cuda" if default_device == "0" else default_device
    if default_label not in device_options:
        default_label = "cpu"

    with st.sidebar:
        st.header("⚙️ 설정")
        device_label = st.selectbox(
            "Device", device_options, index=device_options.index(default_label))
        conf = st.slider("Confidence 임계값", 0.05, 0.95, 0.25, 0.05)
        imgsz = st.select_slider("이미지 크기", options=[480, 640, 832], value=640)
        show_alarm = st.checkbox("위험 알람 오버레이", value=True)
        st.divider()
        st.subheader("위험 클래스")
        for c in HAZARD_CLASSES:
            st.markdown(f"<span class='hazard-pill'>{CLASS_KOR.get(c, c)}</span>",
                        unsafe_allow_html=True)

    # Ultralytics 는 CUDA 디바이스를 "0" 으로 받음.
    device = "0" if device_label == "cuda" else device_label
    detector = load_model(str(WEIGHTS), device)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ 이미지 분석", "🎞️ 영상 분석 + 추적",
        "🗺️ 2D 평면 트윈", "📊 학습 결과",
    ])
    with tab1:
        render_tab_image(detector, conf, imgsz, show_alarm)
    with tab2:
        render_tab_video(detector, conf, imgsz, show_alarm)
    with tab3:
        render_tab_twin()
    with tab4:
        render_tab_metrics()


# Streamlit 은 `streamlit run` 으로 스크립트 전체를 매 상호작용마다 위→아래
# 재실행한다. 가드 없이 최상위에서 한 번 호출하는 것이 올바른 진입점.
main()
