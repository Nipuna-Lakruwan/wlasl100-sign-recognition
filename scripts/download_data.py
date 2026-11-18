"""Download the WLASL100 videos that still exist, plus the MediaPipe landmark models.

Usage:  uv run python scripts/download_data.py

WLASL ships only URLs, and about half of them are dead. The Voxel51/WLASL mirror on
Hugging Face holds the 11,980 clips that survived (all 2,000 words), already trimmed to the
sign and resized to 320x240. We fetch only the clips belonging to the 100-word subset
(1,013 of its 2,038 instances, ~500 MB), pinned to one commit of the mirror.

Writes data/raw/videos/<video_id>.mp4, data/raw/wlasl100.json and models/*.task.
Safe to re-run: existing files are skipped.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WLASL_JSON = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
MIRROR = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/3cf8daaac08088798f539d62fa511028bf5e6fd0"
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def fetch(url: str, dest: Path, tries: int = 5) -> bool:
    """Download url to dest, retrying dropped connections with backoff. Returns False on failure."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:  # listed in the mirror's index but the file is gone: permanent
                print(f"  missing from mirror: {dest.name}", flush=True)
                return False
            print(f"  retry {attempt + 1}/{tries} for {dest.name}: {e}", flush=True)
            time.sleep(2 ** attempt)
        except OSError as e:  # URLError, connection reset, timeouts
            print(f"  retry {attempt + 1}/{tries} for {dest.name}: {e}", flush=True)
            time.sleep(2 ** attempt)
    return False


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    for name, url in MODELS.items():
        fetch(url, ROOT / "models" / name)

    fetch(WLASL_JSON, RAW / "WLASL_v0.3.json")
    fetch(f"{MIRROR}/samples.json", RAW / "voxel51_samples.json")
    wlasl100 = json.loads((RAW / "WLASL_v0.3.json").read_text())[:100]  # WLASL100 = first 100 glosses
    mirror = {Path(s["filepath"]).stem: s["filepath"]
              for s in json.loads((RAW / "voxel51_samples.json").read_text())["samples"]}

    rows, jobs = [], []
    for label, g in enumerate(wlasl100):
        for inst in g["instances"]:
            vid = inst["video_id"]
            path = mirror.get(vid)
            rows.append({"video_id": vid, "gloss": g["gloss"], "label": label, "split": inst["split"],
                         "signer_id": inst["signer_id"], "available": path is not None})
            if path:
                jobs.append((f"{MIRROR}/{path}", RAW / "videos" / f"{vid}.mp4"))
    (RAW / "videos").mkdir(exist_ok=True)
    (RAW / "wlasl100.json").write_text(json.dumps(rows, indent=0))
    print(f"{len(jobs)} of {len(rows)} WLASL100 clips available in the mirror; downloading")

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ok in pool.map(lambda j: fetch(*j), jobs):
            done += 1
            failed += not ok
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    if failed:
        print(f"{failed} clips could not be downloaded (see above)")
    for r in rows:  # what training will actually use
        r["available"] = (RAW / "videos" / f"{r['video_id']}.mp4").exists()
    (RAW / "wlasl100.json").write_text(json.dumps(rows, indent=0))
    size = sum(f.stat().st_size for f in (RAW / "videos").glob("*.mp4"))
    print(f"done: {len(list((RAW / 'videos').glob('*.mp4')))} videos, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
