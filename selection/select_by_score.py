"""Phase 2, minimal version: select a fixed-size training subset from a heterogeneous
pool by a per-document score, then train on it and compare circuit formation.

Pool: induction documents whose hidden per-document repeat fraction is uniform in
[repeat_frac_min, repeat_frac] (`data.induction` with `repeat_frac_min > 0`).

Methods:
- zipper     top-N by per-document LZ77 compression gain (tier 1)
- random     N random documents (baseline)
- refloss    lowest-N mean next-token loss under a reference model trained on a random
             subset of the pool (tier 2; a converged reference has an induction head and
             gives repeat-rich documents low loss)
- oracle     top-N by the hidden per-document repeat fraction (the generator's dial)
- copylen    top-N by the realised copied tokens per document (oracle for what the zipper measures)

    uv run python -m selection.select_by_score --pool datasets/pool_het.npz --method zipper --n 20000 --out datasets/sel_zipper.npz
Writes a dataset with `train` = the selected documents (plus their scores) and `val`
copied from the pool, ready for train.train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.lm_scores import cross_entropy_per_doc
from analysis.zipper import Zipper, strip_pad
from data.generator import load_dataset, save_dataset
from train.train import load_checkpoint


def zipper_scores(tokens: np.ndarray, pad_id: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = Zipper("zlib")
    return np.array([z.compression_gain(strip_pad(d, pad_id), rng) for d in tokens])


def refloss_scores(tokens: np.ndarray, pad_id: int, checkpoint: Path) -> np.ndarray:
    model, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        return cross_entropy_per_doc(model, tokens, pad_id)


def select(
    pool: dict[str, np.ndarray], meta: dict, method: str, n: int, seed: int = 0, reference: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected indices, score array over the pool; higher = preferred)."""
    tokens = pool["tokens"]
    if method == "random":
        score = np.random.default_rng(seed).random(len(tokens))
    elif method == "zipper":
        score = zipper_scores(tokens, meta["pad_id"], seed)
    elif method == "refloss":
        if reference is None:
            raise ValueError("refloss needs --reference checkpoint")
        score = -refloss_scores(tokens, meta["pad_id"], reference)
    elif method == "oracle":
        score = pool["doc_frac"].astype(float)
    elif method == "copylen":  # oracle for the realised copy length (what the zipper appears to measure)
        score = pool["doc_copied"].astype(float) + 1e-3 * np.random.default_rng(seed).random(len(tokens))
    else:
        raise ValueError(method)
    idx = np.argsort(-score, kind="stable")[:n]
    return np.sort(idx), score


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--method", required=True, choices=["zipper", "random", "refloss", "oracle", "copylen"])
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference", type=Path, default=None, help="checkpoint for refloss")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    splits, meta = load_dataset(a.pool)
    idx, score = select(splits["train"], meta, a.method, a.n, a.seed, a.reference)
    train = {k: v[idx] for k, v in splits["train"].items()}
    train["score"] = score[idx]
    meta = dict(meta)
    meta["selection"] = {"method": a.method, "n": a.n, "seed": a.seed, "pool": str(a.pool), "mean_score_selected": float(score[idx].mean()), "mean_score_pool": float(score.mean())}
    if "doc_frac" in splits["train"]:  # synthetic pools carry the hidden per-document fraction
        frac = splits["train"]["doc_frac"]
        meta["selection"].update(
            mean_doc_frac_selected=float(frac[idx].mean()), mean_doc_frac_pool=float(frac.mean()),
            spearman_score_vs_hidden_frac=float(_spearman(score, frac)),
        )
    meta["stats"] = dict(meta.get("stats", {}), selected={"frac_target": float(train["target_mask"].mean())})
    save_dataset(a.out, {"train": train, "val": splits["val"]}, meta)
    print(json.dumps(meta["selection"]))
    return a.out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    main()
