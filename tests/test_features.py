import numpy as np
import torch

from sign.landmarks import FEAT_DIM, normalise
from sign.model import T, SignGRU, resample, trim_to_hands


def _pose(offset=(0.0, 0.0), scale=1.0):
    base = np.array([[0, -1], [-1, 0], [1, 0], [-1.5, 1], [1.5, 1], [-1, 2], [1, 2]], dtype=np.float32)
    return base * scale + np.array(offset, dtype=np.float32)


def test_normalise_ignores_position_and_distance():
    hand = np.tile(np.array([[-1, 2]], dtype=np.float32), (21, 1))
    a = normalise(_pose(), [hand])
    b = normalise(_pose((300, 120), 4.0), [hand * 4.0 + np.array([300, 120], dtype=np.float32)])
    assert a.shape == (FEAT_DIM,)
    assert np.allclose(a, b, atol=1e-5)


def test_hand_goes_to_nearest_wrist_slot():
    near_left = np.tile(np.array([[-1, 2]], dtype=np.float32), (21, 1))
    near_right = np.tile(np.array([[1, 2]], dtype=np.float32), (21, 1))
    assert list(normalise(_pose(), [near_left])[-2:]) == [1, 0]
    assert list(normalise(_pose(), [near_right])[-2:]) == [0, 1]
    assert list(normalise(_pose(), [near_right, near_left])[-2:]) == [1, 1]


def test_missing_pose_gives_zeros():
    assert not normalise(None, []).any()


def test_resample_and_trim():
    seq = np.zeros((50, FEAT_DIM), dtype=np.float32)
    seq[20:30, -1] = 1  # hand visible only in frames 20-29
    assert len(trim_to_hands(seq)) == 14  # 10 frames + 2 padding each side
    assert resample(seq).shape == (T, FEAT_DIM)
    assert resample(seq[:5]).shape == (T, FEAT_DIM)


def test_model_output_shape():
    out = SignGRU()(torch.zeros(3, T, FEAT_DIM))
    assert out.shape == (3, 100)


def test_signer_split_keeps_each_signer_in_one_part():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import signer_split

    signer = np.repeat(np.arange(30), np.arange(1, 31))  # 30 signers with 1..30 clips each
    sp = signer_split(signer)
    for s in np.unique(signer):
        assert len(set(sp[signer == s])) == 1
    assert {"train", "val", "test"} <= set(sp)
