"""Download a warehouse / industrial-safety detection dataset from Roboflow Universe.

We try a small ranked list of well-known public datasets and stop at the first
one that downloads cleanly. Output goes into app/datasets/<dataset_name>/.

Usage (from the app/ directory):
    ROBOFLOW_API_KEY=xxxxx uv run python scripts/get_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from roboflow import Roboflow


# (workspace, project_slug, version, label_for_logging)
# Ranked by relevance to warehouse / industrial CV (person + machinery + PPE).
CANDIDATES = [
    ("roboflow-universe-projects", "construction-site-safety", 27, "construction-ppe"),
    ("roboflow-100", "construction-safety-gsnvb", 1, "construction-safety-r100"),
    ("titulacin", "personal-protective-equipment", 4, "ppe-titulacin"),
    ("forklift-bxnq5", "forklift-detection-eumlx", 2, "forklift-bxnq5"),
]

# Anchor to app/ root (this script lives in app/scripts/) so the dataset lands
# where the training scripts expect it, regardless of the current directory.
OUT_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("[error] set ROBOFLOW_API_KEY env var")

    rf = Roboflow(api_key=api_key)
    OUT_ROOT.mkdir(exist_ok=True)

    last_err: Exception | None = None
    for workspace, slug, version, label in CANDIDATES:
        target = OUT_ROOT / label
        if (target / "data.yaml").exists():
            print(f"[skip] already downloaded -> {target}")
            print(f"[done] using existing: {target / 'data.yaml'}")
            return
        try:
            print(f"[try] {workspace}/{slug} v{version} -> {label}")
            project = rf.workspace(workspace).project(slug)
            ds = project.version(version).download(
                "yolov11",
                location=str(target),
                overwrite=False,
            )
            print(f"[ok]  downloaded -> {ds.location}")
            print(f"[done] data.yaml at: {Path(ds.location) / 'data.yaml'}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {workspace}/{slug}: {exc}")
            last_err = exc

    sys.exit(f"[error] no candidate worked. last error: {last_err}")


if __name__ == "__main__":
    main()
