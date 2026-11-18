"""MediaPipe landmarks -> one fixed-size feature vector per video frame.

Used by BOTH the offline extraction (training data) and the live webcam demo, so the model
always sees features computed exactly the same way.

Per frame we keep 49 points (x, y):
  * 7 upper-body pose points: nose, both shoulders, elbows, wrists
  * 21 points for each hand (left slot, right slot)
and normalise them so the signer's position and distance from the camera don't matter:
origin = midpoint of the shoulders, unit = shoulder width.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_IDS = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist
N_POINTS = len(POSE_IDS) + 21 + 21      # 49
FEAT_DIM = N_POINTS * 2 + 2             # xy for each point + 2 hand-present flags = 100
MODELS = Path(__file__).resolve().parents[2] / "models"


def normalise(pose: np.ndarray | None, hands: list[np.ndarray]) -> np.ndarray:
    """pose: (7, 2) pixel coords or None. hands: up to two (21, 2) pixel-coord arrays.

    Hands are put in the left/right slot by which pose wrist they are closest to (more
    reliable than MediaPipe's handedness label, which depends on whether the image is mirrored).
    Returns a (FEAT_DIM,) float32 vector; missing parts are zeros with a 0 presence flag.
    """
    out = np.zeros(FEAT_DIM, dtype=np.float32)
    if pose is None:
        return out
    centre = (pose[1] + pose[2]) / 2
    scale = np.linalg.norm(pose[1] - pose[2]) + 1e-6
    pts = np.zeros((N_POINTS, 2), dtype=np.float32)
    pts[:7] = (pose - centre) / scale
    present = [0.0, 0.0]
    for hand in hands:
        wrist = hand[0]
        slot = 0 if np.linalg.norm(wrist - pose[5]) <= np.linalg.norm(wrist - pose[6]) else 1
        if present[slot]:  # both hands nearest the same wrist: put the second in the free slot
            slot = 1 - slot
            if present[slot]:
                continue
        pts[7 + 21 * slot: 7 + 21 * (slot + 1)] = (hand - centre) / scale
        present[slot] = 1.0
    out[:-2] = pts.reshape(-1)
    out[-2:] = present
    return out


class Extractor:
    """Wraps MediaPipe's HandLandmarker and PoseLandmarker in VIDEO mode (they track across frames)."""

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        mode = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
            running_mode=mode, num_poses=1))

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, dict]:
        """rgb: HxWx3 uint8. Timestamps must increase. Returns (feature vector, raw points for drawing)."""
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pr = self.pose.detect_for_video(image, timestamp_ms)
        hr = self.hands.detect_for_video(image, timestamp_ms)
        to_px = lambda lms: np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)  # noqa: E731
        pose = to_px([pr.pose_landmarks[0][i] for i in POSE_IDS]) if pr.pose_landmarks else None
        hands = [to_px(h_) for h_ in hr.hand_landmarks]
        return normalise(pose, hands), {"pose": pose, "hands": hands}

    def close(self) -> None:
        self.hands.close()
        self.pose.close()


def video_features(path: Path) -> np.ndarray:
    """All frames of a video file -> (n_frames, FEAT_DIM). A fresh Extractor per video, so tracking restarts."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ex = Extractor()
    feats, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f, _ = ex(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(i * 1000 / fps))
        feats.append(f)
        i += 1
    cap.release()
    ex.close()
    return np.stack(feats) if feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
