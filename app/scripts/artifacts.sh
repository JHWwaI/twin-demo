#!/usr/bin/env bash
# twin-demo 무거운 산출물(footage / datasets / 학습 weights) 관리.
#
# 위치:
#   _artifacts/twin-demo/footage/                          # 영상 (~200MB)
#   _artifacts/twin-demo/datasets/construction-ppe/        # 학습 데이터 (~165MB)
#   _artifacts/twin-demo/runs/detect/runs/train/exp01_*    # 학습 weights + 그래프 (~85MB)
#
# 사용:
#   bash scripts/artifacts.sh link     # _artifacts → twin-demo 심볼릭 링크 생성 (Streamlit 실행 가능 상태)
#   bash scripts/artifacts.sh unlink   # 심볼릭 링크 제거 (커밋 직전 깨끗한 상태)
#   bash scripts/artifacts.sh status   # 현재 상태 확인
#   bash scripts/artifacts.sh sync     # _artifacts 가 없으면 안내 (다른 PC로 옮길 때)
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"           # twin-demo/
ART="$(cd "$HERE/.." && pwd)/_artifacts/twin-demo"                # _artifacts/twin-demo/

PAIRS=(
  "footage::$ART/footage"
  "datasets::$ART/datasets"
  "runs/detect/runs/train::$ART/runs/detect/runs/train"
)

resolve_target() {
  echo "$1" | awk -F '::' '{print $2}'
}
resolve_link() {
  echo "$HERE/$(echo "$1" | awk -F '::' '{print $1}')"
}

cmd_link() {
  if [ ! -d "$ART" ]; then
    echo "[error] _artifacts 폴더가 없습니다: $ART"
    echo "  다른 PC에서 처음 클론한 경우 _artifacts.tar.gz 를 받아 풀어주세요."
    exit 1
  fi
  for p in "${PAIRS[@]}"; do
    LINK="$(resolve_link "$p")"
    TARGET="$(resolve_target "$p")"
    mkdir -p "$(dirname "$LINK")"
    if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
      echo "  [skip] $LINK 가 실제 디렉토리로 존재 — 수동 확인 필요"
      continue
    fi
    rm -f "$LINK"
    ln -s "$TARGET" "$LINK"
    echo "  link: $LINK -> $TARGET"
  done
  echo "[done] Streamlit 실행 가능 상태"
}

cmd_unlink() {
  for p in "${PAIRS[@]}"; do
    LINK="$(resolve_link "$p")"
    if [ -L "$LINK" ]; then
      rm "$LINK"
      echo "  unlink: $LINK"
    fi
  done
  echo "[done] 커밋 가능한 깨끗한 상태"
}

cmd_status() {
  echo "ART: $ART"
  [ -d "$ART" ] && echo "  존재 (크기: $(du -sh "$ART" | cut -f1))" || echo "  없음"
  echo ""
  for p in "${PAIRS[@]}"; do
    LINK="$(resolve_link "$p")"
    if [ -L "$LINK" ]; then
      echo "  [link]   $LINK -> $(readlink "$LINK")"
    elif [ -d "$LINK" ]; then
      echo "  [dir]    $LINK ($(du -sh "$LINK" | cut -f1))"
    else
      echo "  [none]   $LINK"
    fi
  done
}

cmd_sync() {
  if [ -d "$ART" ]; then
    echo "_artifacts 있음. 'bash scripts/artifacts.sh link' 로 연결하세요."
    return
  fi
  cat <<EOF
[다른 PC로 옮긴 후 처음 실행하는 경우]

_artifacts/twin-demo/ 가 없습니다. 3가지 방법 중 하나:

1) 클라우드/외장 디스크에서 _artifacts.tar.gz 받아 풀기:
     cd $(cd "$HERE/.." && pwd)
     tar -xzf _artifacts.tar.gz
     bash $HERE/scripts/artifacts.sh link

2) 원격(scp 등)으로 받기:
     scp user@host:/path/to/_artifacts.tar.gz .
     tar -xzf _artifacts.tar.gz
     bash $HERE/scripts/artifacts.sh link

3) 처음부터 다시 받기 (Roboflow):
     # ROBOFLOW_API_KEY 환경변수 설정 후
     uv run python scripts/get_dataset.py   # datasets/construction-ppe 자동 생성
     # 영상은 본인 footage 또는 데모용 외부 영상 사용
     # 학습:  uv run python scripts/train_baseline.py   (1-3시간)

학습된 best.pt 없으면 Streamlit Tab 1~3 이 동작하지 않습니다.
검증 산출물(ppe_eval)은 코드와 함께 보관되어 Tab 4 는 동작합니다.
EOF
}

case "${1:-help}" in
  link)   cmd_link ;;
  unlink) cmd_unlink ;;
  status) cmd_status ;;
  sync)   cmd_sync ;;
  *) echo "사용: bash $0 {link|unlink|status|sync}" ;;
esac
