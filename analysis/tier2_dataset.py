"""Tier-2 dataset score: mean next-token cross-entropy of a dataset's validation documents
under a fixed reference model (one trained on a mixed pool that has formed an induction
head). Lower loss = more structure the reference can exploit. Reported as a gain in nats
relative to the loss on shuffled copies of the same documents, so unigram statistics do
not count (same construction as the zipper gain).

    uv run python -m analysis.tier2_dataset --reference checkpoints/sel_random_s0/step_6000.pt --runs ind_rep0.5_s0 ind_vocab16_s0 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.lm_scores import cross_entropy_per_doc
from data.generator import load_dataset
from train.train import load_checkpoint


def tier2_gain(model, tokens: np.ndarray, pad_id: int, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    ce = cross_entropy_per_doc(model, tokens, pad_id).mean()
    shuffled = tokens.copy()
    for i in range(len(shuffled)):
        body = shuffled[i, 1:]
        shuffled[i, 1:] = body[rng.permutation(len(body))]
    ce_shuf = cross_entropy_per_doc(model, shuffled, pad_id).mean()
    return float(ce), float(ce_shuf), float(1.0 - ce / ce_shuf)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--n-docs", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("results/tier2_dataset_scores.csv"))
    a = p.parse_args(argv)
    model, _ = load_checkpoint(a.reference)
    model.eval()
    rows = []
    with torch.no_grad():
        for run in a.runs:
            tc = json.loads(Path("results", run, "train_config.json").read_text())
            splits, meta = load_dataset(Path(tc["dataset"]))
            ce, ce_shuf, gain = tier2_gain(model, splits["val"]["tokens"][: a.n_docs], meta["pad_id"])
            rows.append({"run": run, "tier2_ce": ce, "tier2_ce_shuffled": ce_shuf, "tier2_gain": gain, **{k: v for k, v in meta["config"].items() if k in ("repeat_frac", "noise", "vocab_size", "n_repeats")}})
            print(f"{run:22s} ce {ce:.3f} shuffled {ce_shuf:.3f} tier2_gain {gain:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    return df


if __name__ == "__main__":
    main()
