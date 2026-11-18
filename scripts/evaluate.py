"""Score the trained models on the WLASL100 test clips and make the README plots.

Usage:  uv run python scripts/evaluate.py

Headline (fixed before scoring test): average of the seed models' probabilities (gru_s0..s2).
Reports top-1 / top-5 accuracy, splits test clips by whether their signer also appears in
training (WLASL's official split is not signer-independent), and lists the most confused pairs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sign.model import SignGRU  # noqa: E402
from train import batch_eval, load, signer_split  # noqa: E402

SEEDS = ["gru_s0", "gru_s1", "gru_s2"]
SIGNER_SEEDS = ["gru_s0_signer", "gru_s1_signer", "gru_s2_signer"]


def wilson(p: float, n: int, z: float = 1.96) -> list[float]:
    """95% confidence interval for an accuracy measured on n examples (Wilson score interval)."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 3), round(float(centre + half), 3)]


def topk(p: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-p, 1)[:, :k] == y[:, None]).any(1).mean())


def main() -> None:
    d = load()
    gloss = [str(g) for g in d["gloss"]]
    te = np.where(d["split"] == "test")[0]
    clips, y = [d["clips"][i] for i in te], d["label"][te]

    probs, res = {}, {}
    for name in SEEDS + ["gru_s0_noaug"]:
        path = ROOT / "checkpoints" / f"{name}.pt"
        if not path.exists():
            continue
        m = SignGRU()
        m.load_state_dict(torch.load(path, map_location="cpu")["model"])
        probs[name] = batch_eval(m, clips)
        res[name] = {"top1": round(topk(probs[name], y, 1), 3), "top5": round(topk(probs[name], y, 5), 3)}
    ens = np.mean([probs[n] for n in SEEDS if n in probs], 0)
    res["headline_ensemble"] = {"top1": round(topk(ens, y, 1), 3), "top5": round(topk(ens, y, 5), 3),
                                "top10": round(topk(ens, y, 10), 3)}

    res["headline_ensemble"]["top1_95ci"] = wilson(res["headline_ensemble"]["top1"], len(te))

    # Signer-independent experiment: whole signers held out of train and val (train.py --split signer)
    sp = signer_split(d["signer"])
    te_s = np.where(sp == "test")[0]
    clips_s, y_s = [d["clips"][i] for i in te_s], d["label"][te_s]
    ps = []
    for n in SIGNER_SEEDS:
        m = SignGRU()
        m.load_state_dict(torch.load(ROOT / "checkpoints" / f"{n}.pt", map_location="cpu")["model"])
        ps.append(batch_eval(m, clips_s))
        res[n] = {"top1": round(topk(ps[-1], y_s, 1), 3), "top5": round(topk(ps[-1], y_s, 5), 3)}
    ens_s = np.mean(ps, 0)
    t1 = round(topk(ens_s, y_s, 1), 3)
    res["signer_independent_ensemble"] = {
        "top1": t1, "top1_95ci": wilson(t1, len(te_s)), "top5": round(topk(ens_s, y_s, 5), 3),
        "top10": round(topk(ens_s, y_s, 10), 3), "test_clips": len(te_s),
        "test_signers": len(set(d["signer"][te_s].tolist())), "words_with_test_clips": int(len(np.unique(y_s)))}

    train_signers = set(d["signer"][d["split"] == "train"].tolist())
    seen = np.array([s in train_signers for s in d["signer"][te]])
    pred = ens.argmax(1)
    res["by_signer"] = {
        "signer_also_in_train": {"clips": int(seen.sum()), "top1": round(float((pred[seen] == y[seen]).mean()), 3)},
        "new_signer": {"clips": int((~seen).sum()), "top1": round(float((pred[~seen] == y[~seen]).mean()), 3)},
    }
    # Test clips per word: at most 3, so per-word accuracy is very noisy; report the pairs instead.
    confusions = Counter((gloss[a], gloss[b]) for a, b in zip(y, pred) if a != b)
    res["most_common_confusions"] = [f"{a} -> {b} (x{n})" for (a, b), n in confusions.most_common(10)]
    res["test_clips"] = len(te)
    res["words_with_test_clips"] = int(len(np.unique(y)))
    res["chance_top1"] = round(1 / 100, 3)
    hand_frac = np.array([c[:, -2:].max(1).mean() for c in clips])
    right = pred == y
    res["hand_visible_fraction"] = {"correct": round(float(hand_frac[right].mean()), 3),
                                    "wrong": round(float(hand_frac[~right].mean()), 3)}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # Plots: validation curves; top-k curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for n in SEEDS + ["gru_s0_noaug"]:
        f = ROOT / "results" / f"{n}_history.json"
        if f.exists():
            h = json.loads(f.read_text())["history"]
            axes[0].plot([e["epoch"] for e in h], [e["val_top1"] for e in h], label=n, lw=1)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation top-1")
    axes[0].legend(fontsize=8)
    ks = np.arange(1, 21)
    axes[1].plot(ks, [topk(ens, y, k) for k in ks], marker="o", ms=3)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("test top-k accuracy")
    axes[1].set_xticks([1, 5, 10, 15, 20])
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "plots" / "curves.png", dpi=120)
    plt.close(fig)

    # A correct and a wrong example: hand trajectories (right-hand wrist path)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    picks = list(np.where(right)[0][:2]) + list(np.where(~right)[0][:2])
    for ax, i in zip(axes, picks):
        c = clips[i]
        for slot, col in ((0, "tab:blue"), (1, "tab:orange")):
            on = c[:, -2 + slot] > 0
            base = 7 + 21 * slot
            ax.plot(c[on, 2 * base], -c[on, 2 * base + 1], color=col, lw=1, marker=".", ms=3)
        ax.plot(c[0, 2:6:2], -c[0, 3:7:2], "k-", lw=2)  # shoulders
        ax.set_title(f"true: {gloss[y[i]]}\npred: {gloss[pred[i]]}", fontsize=9,
                     color="green" if right[i] else "red")
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Wrist paths of the two hand slots over a clip (black = shoulders)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "plots" / "examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
