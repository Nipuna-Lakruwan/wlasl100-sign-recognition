"""Run MediaPipe once over every downloaded clip and cache the landmark features.

Usage:  uv run python scripts/extract_landmarks.py [--workers 3]

Writes data/processed/landmarks/<video_id>.npy (n_frames x 100), then packs everything into
data/processed/wlasl100_landmarks.npz (~10 MB). After this, the videos are no longer needed
for training. Safe to re-run: finished clips are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "processed" / "landmarks"


def work(vid: str) -> tuple[str, int]:
    os.environ.setdefault("GLOG_minloglevel", "2")
    from sign.landmarks import video_features

    dest = OUT / f"{vid}.npy"
    video = ROOT / "data" / "raw" / "videos" / f"{vid}.mp4"
    if not dest.exists():
        if not video.exists():
            return vid, -1  # not downloaded yet
        np.save(dest, video_features(video))
    return vid, len(np.load(dest))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((ROOT / "data" / "raw" / "wlasl100.json").read_text()) if r["available"]]
    t0 = time.time()
    missing = 0
    with Pool(args.workers) as pool:
        for k, (vid, n) in enumerate(pool.imap_unordered(work, [r["video_id"] for r in rows]), 1):
            missing += n < 0
            if k % 50 == 0:
                print(f"{k}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    if missing:
        print(f"{missing} videos not downloaded yet; run again after download_data.py finishes")
        return

    feats = [np.load(OUT / f"{r['video_id']}.npy") for r in rows]
    lengths = np.array([len(f) for f in feats])
    np.savez_compressed(
        ROOT / "data" / "processed" / "wlasl100_landmarks.npz",
        feats=np.concatenate(feats), lengths=lengths,
        label=np.array([r["label"] for r in rows]), split=np.array([r["split"] for r in rows]),
        signer=np.array([r["signer_id"] for r in rows]), video_id=np.array([r["video_id"] for r in rows]),
        gloss=np.array([g for g in dict.fromkeys(r["gloss"] for r in json.loads(
            (ROOT / "data" / "raw" / "wlasl100.json").read_text()))]),
    )
    hands = np.concatenate(feats)[:, -2:].max(1)
    print(f"packed {len(rows)} clips, {lengths.sum()} frames; frames with >=1 hand: {hands.mean():.1%}; "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
