"""Two-quantity selectors for a real pool: gain for formation, diversity for coverage.

- mix      half the budget by gzip gain (top-N/2), half uniformly at random from the rest
- divgain  greedy by gain with a diversity constraint: walk documents in descending gain,
           accept one only if its NCD to each of `--probe` randomly drawn already-accepted
           documents exceeds `--ncd-min`; stop at N

    uv run python -m selection.two_quantity --pool datasets/code_pool.npz --method mix --n 100000 --seed 0 --out datasets/code_big_mix_s0.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.zipper import ncd, strip_pad, to_bytes
from data.generator import load_dataset, save_dataset
from selection.select_by_score import zipper_scores


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--method", choices=["mix", "divgain"], required=True)
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncd-min", type=float, default=0.75)
    p.add_argument("--probe", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    splits, meta = load_dataset(a.pool)
    tokens = splits["train"]["tokens"]
    pad = meta["pad_id"]
    rng = np.random.default_rng(a.seed)
    cache = Path(a.pool).with_suffix(".gain.npy")
    gain = np.load(cache) if cache.exists() else zipper_scores(tokens, pad)
    if not cache.exists():
        np.save(cache, gain)
    order = np.argsort(-gain, kind="stable")
    if a.method == "mix":
        top = order[: a.n // 2]
        rest = np.setdiff1d(np.arange(len(tokens)), top)
        idx = np.concatenate([top, rng.choice(rest, a.n - len(top), replace=False)])
        info = {"n_by_gain": int(len(top)), "n_random": int(a.n - len(top))}
    else:
        docs = [to_bytes(strip_pad(t, pad)) for t in tokens]
        chosen: list[int] = []
        rejected = 0
        for i in order:
            if len(chosen) >= a.n:
                break
            if chosen:
                sample = rng.choice(len(chosen), min(a.probe, len(chosen)), replace=False)
                if min(ncd(docs[i], docs[chosen[j]]) for j in sample) < a.ncd_min:
                    rejected += 1
                    continue
            chosen.append(int(i))
        idx = np.array(chosen)
        info = {"rejected": rejected, "ncd_min": a.ncd_min, "probe": a.probe, "lowest_gain_accepted": float(gain[idx].min())}
    idx = np.sort(idx)
    train = {k: v[idx] for k, v in splits["train"].items()}
    train["score"] = gain[idx]
    meta = dict(meta)
    meta["selection"] = {"method": a.method, "n": int(len(idx)), "seed": a.seed, "pool": str(a.pool), "mean_score_selected": float(gain[idx].mean()), "mean_score_pool": float(gain.mean()), **info}
    save_dataset(a.out, {"train": train, "val": splits["val"]}, meta)
    print(json.dumps(meta["selection"]))
    return a.out


if __name__ == "__main__":
    main()
