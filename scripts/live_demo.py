"""Live sign recognition from the webcam (or a video file).

Usage:
  uv run python scripts/live_demo.py                    # webcam 0; macOS asks for camera access the first time
  uv run python scripts/live_demo.py --video clip.mp4   # same pipeline on a file (no camera needed)
  uv run python scripts/live_demo.py --video clip.mp4 --headless   # print predictions only

How to sign for it: start with your hands down, raise them, sign ONE word, lower your hands.
While your hands are up, frames are collected. When they've been down for ~0.3 s, the
collected segment is classified and the word appears at the top. Press q to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sign.landmarks import Extractor  # noqa: E402
from sign.model import SignGRU, resample, trim_to_hands  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Recogniser:
    """Hands up -> collect frames; hands down for `gap` frames -> classify the segment."""

    def __init__(self, ckpt: Path, glosses: list[str], gap: int = 8, min_len: int = 10):
        self.model = SignGRU()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.eval()
        self.glosses, self.gap, self.min_len = glosses, gap, min_len
        self.buf: list[np.ndarray] = []
        self.missing = 0

    def step(self, feat: np.ndarray) -> list[tuple[str, float]] | None:
        hands_up = feat[-2:].max() > 0
        if hands_up:
            self.buf.append(feat)
            self.missing = 0
            return None
        if not self.buf:
            return None
        self.buf.append(feat)
        self.missing += 1
        if self.missing < self.gap:
            return None
        seg, self.buf, self.missing = np.stack(self.buf), [], 0
        seg = trim_to_hands(seg)
        if len(seg) < self.min_len:
            return None
        return self.predict(seg)

    def flush(self) -> list[tuple[str, float]] | None:
        if len(self.buf) >= self.min_len:
            seg, self.buf = trim_to_hands(np.stack(self.buf)), []
            return self.predict(seg)
        return None

    def predict(self, seg: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(resample(seg))[None]), -1)[0].numpy()
        top = np.argsort(-p)[:k]
        return [(self.glosses[i], float(p[i])) for i in top]


def draw(frame: np.ndarray, pts: dict) -> None:
    for hand in pts["hands"]:
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(map(int, hand[a])), tuple(map(int, hand[b])), (0, 220, 120), 2)
    if pts["pose"] is not None:
        for x, y in pts["pose"]:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 160, 0), -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "gru_s0.pt")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    glosses = [str(g) for g in np.load(ROOT / "data" / "processed" / "wlasl100_landmarks.npz")["gloss"]]
    rec = Recogniser(args.checkpoint, glosses)
    ex = Extractor()
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera. On macOS: System Settings > Privacy & Security > Camera, "
                 "allow your terminal app, then run again.")
    fps_file = cap.get(cv2.CAP_PROP_FPS) or 25.0
    last, t_start, i = None, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000 / fps_file) if args.video else int((time.time() - t_start) * 1000)
        feat, pts = ex(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)  # model sees the un-mirrored frame
        i += 1
        result = rec.step(feat)
        if result:
            last = result
            print("prediction:", ", ".join(f"{g} {p:.0%}" for g, p in result), flush=True)
        if args.headless:
            continue
        draw(frame, pts)
        view = cv2.flip(frame, 1) if not args.video else frame  # mirror the webcam like a selfie camera
        status = "collecting..." if rec.buf else "raise hands, sign one word, lower hands"
        cv2.putText(view, status, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if last:
            cv2.rectangle(view, (0, 0), (view.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(view, last[0][0].upper(), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            alts = "   ".join(f"{g} {p:.0%}" for g, p in last)
            cv2.putText(view, alts, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("WLASL100 sign recognition (q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    final = rec.flush()
    if final:
        print("prediction:", ", ".join(f"{g} {p:.0%}" for g, p in final))
    cap.release()
    ex.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
