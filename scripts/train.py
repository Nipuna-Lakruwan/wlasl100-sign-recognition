"""Train the GRU on cached WLASL100 landmarks.

Usage:  uv run python scripts/train.py [--epochs 200] [--seed 0] [--no-aug]

Uses WLASL's official train/val/test split. Picks the epoch with the best validation top-1
accuracy; never looks at test. Saves checkpoints/<name>.pt and results/<name>_history.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sign.landmarks import N_POINTS  # noqa: E402
from sign.model import T, SignGRU, resample, trim_to_hands  # noqa: E402

# Mirror image: swap left/right pose points and the two hand slots.
_FLIP = np.arange(N_POINTS)
_FLIP[[1, 2, 3, 4, 5, 6]] = [2, 1, 4, 3, 6, 5]
_FLIP[7:28], _FLIP[28:49] = np.arange(28, 49), np.arange(7, 28)


def signer_split(signer: np.ndarray, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 123) -> np.ndarray:
    """Assign whole signers to test (~20% of clips), val (~10%) and train, ignoring WLASL's split.

    WLASL's official split is by clip: 97 of the 100 surviving test clips are by people who also
    appear in training. This split answers "how well does it work for someone it has never seen?"
    The signer order is a fixed shuffle (seed 123), chosen once, independent of results.
    """
    ids, counts = np.unique(signer, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(ids))
    out = np.full(len(signer), "train", dtype=object)
    total, acc = len(signer), 0
    for k in order:
        acc += counts[k]
        part = "test" if acc <= test_frac * total else "val" if acc <= (test_frac + val_frac) * total else "train"
        out[signer == ids[k]] = part
    return out.astype(str)


def load() -> dict:
    d = dict(np.load(ROOT / "data" / "processed" / "wlasl100_landmarks.npz"))
    ends = np.cumsum(d["lengths"])
    d["clips"] = [trim_to_hands(c) for c in np.split(d["feats"], ends[:-1])]
    return d


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make one random variant of a clip: different speed/crop, size, angle, position, handedness."""
    n = len(seq)
    if n > 8:  # random temporal crop of 80-100% of the clip
        keep = int(n * rng.uniform(0.8, 1.0))
        s = rng.integers(0, n - keep + 1)
        seq = seq[s:s + keep]
    seq = resample(seq).copy()
    pts = seq[:, :-2].reshape(T, N_POINTS, 2)
    flags = seq[:, -2:].copy()
    if rng.random() < 0.5:  # mirror: a left-handed signer
        pts = pts[:, _FLIP] * np.array([-1, 1], dtype=np.float32)
        flags = flags[:, ::-1]
    a = np.deg2rad(rng.uniform(-15, 15))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    scale = rng.uniform(0.85, 1.15, size=2).astype(np.float32)
    shift = rng.normal(0, 0.05, size=2).astype(np.float32)
    nonzero = np.abs(pts).sum(-1, keepdims=True) > 0  # don't move missing points away from 0
    pts = np.where(nonzero, (pts @ rot.T) * scale + shift + rng.normal(0, 0.01, pts.shape), 0)
    return np.concatenate([pts.reshape(T, -1), flags], axis=1).astype(np.float32)


def batch_eval(model: nn.Module, clips: list[np.ndarray]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.stack([resample(c) for c in clips]))
        return torch.softmax(model(x), -1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", choices=["official", "signer"], default="official",
                    help="signer: hold out whole signers for val and test (see signer_split)")
    args = ap.parse_args()
    name = args.name or f"gru_s{args.seed}{'_noaug' if args.no_aug else ''}{'_signer' if args.split == 'signer' else ''}"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d = load()
    split = signer_split(d["signer"]) if args.split == "signer" else d["split"]
    idx = {s: np.where(split == s)[0] for s in ("train", "val")}
    tr = [d["clips"][i] for i in idx["train"]]
    ytr = torch.from_numpy(d["label"][idx["train"]])
    va, yva = [d["clips"][i] for i in idx["val"]], d["label"][idx["val"]]
    print(f"[{name}] train {len(tr)} clips, val {len(va)} clips, 100 classes")

    # The GRU is small; on this machine the CPU is as fast as MPS for it and avoids MPS GRU quirks.
    model = SignGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * ((len(tr) + 31) // 32))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, history, t0 = -1.0, [], time.time()
    ckpt = ROOT / "checkpoints" / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0.0
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            xb = np.stack([resample(tr[k]) if args.no_aug else augment(tr[k], rng) for k in j])
            loss = loss_fn(model(torch.from_numpy(xb)), ytr[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(j)
        p = batch_eval(model, va)
        top1 = float((p.argmax(1) == yva).mean())
        top5 = float((np.argsort(-p, 1)[:, :5] == yva[:, None]).any(1).mean())
        history.append({"epoch": epoch, "train_loss": total / len(tr), "val_top1": top1, "val_top5": top5})
        flag = ""
        if top1 > best:
            best, flag = top1, " *"
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed, "split": args.split}, ckpt)
        if epoch % 10 == 0 or flag:
            print(f"epoch {epoch:3d}  loss {total / len(tr):.3f}  val top1 {top1:.3f} top5 {top5:.3f}{flag}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    (ROOT / "results" / f"{name}_history.json").write_text(
        json.dumps({"history": history, "best_val_top1": best, "minutes": (time.time() - t0) / 60}, indent=1))
    print(f"best val top-1 {best:.3f}")


if __name__ == "__main__":
    main()
