"""Sequence model over landmark frames, and the preprocessing it expects."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sign.landmarks import FEAT_DIM, N_POINTS

T = 32  # every clip is resampled to 32 frames


def resample(seq: np.ndarray, t: int = T) -> np.ndarray:
    """Pick t evenly spaced frames (nearest neighbour). Works for clips shorter or longer than t."""
    if len(seq) == 0:
        return np.zeros((t, seq.shape[1]), dtype=np.float32)
    idx = np.linspace(0, len(seq) - 1, t).round().astype(int)
    return seq[idx]


def trim_to_hands(seq: np.ndarray, pad: int = 2) -> np.ndarray:
    """Drop the frames before the first and after the last frame with a visible hand.

    Signs happen while the hands are up; the rest is the signer standing still. The live demo
    uses the same rule to decide which frames to classify.
    """
    hand = seq[:, -2:].max(1) > 0
    if not hand.any():
        return seq
    i, j = np.where(hand)[0][[0, -1]]
    return seq[max(0, i - pad): j + pad + 1]


def add_velocity(x: torch.Tensor) -> torch.Tensor:
    """(B, T, F) -> (B, T, 2F - 2): positions plus frame-to-frame motion of every point."""
    pos = x[..., :-2]
    vel = torch.diff(pos, dim=1, prepend=pos[:, :1])
    return torch.cat([pos, vel, x[..., -2:]], dim=-1)


class SignGRU(nn.Module):
    """Per-frame MLP -> 2-layer bidirectional GRU -> mean+max pooling over time -> 100 classes."""

    def __init__(self, n_classes: int = 100, hidden: int = 128, dropout: float = 0.4):
        super().__init__()
        f_in = 2 * N_POINTS * 2 + 2
        self.frame = nn.Sequential(nn.Linear(f_in, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(192, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, FEAT_DIM)
        h, _ = self.gru(self.frame(add_velocity(x)))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


assert FEAT_DIM == N_POINTS * 2 + 2
