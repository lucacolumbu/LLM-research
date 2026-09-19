"""Name-swap control for the in-context delta: replace the first occurrence of a rare
identifier x with a different rare identifier y (absent from the document) and measure, at
the second occurrence, log p(x) and log p(y) before and after the swap. A load-bearing copy
head moves probability from x to y; syntax-based prediction does not.

    uv run python -m analysis.swap_control --step 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.incontext_delta import name_ranks, tag_documents
from data.generator import load_dataset
from train.train import load_checkpoint


@torch.no_grad()
def gather_logprobs(model, tokens: np.ndarray, positions: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray, batch: int = 64):
    """log p(ids_a) and log p(ids_b) at the prediction position for each document, gathered
    per batch so the full vocabulary log-softmax is never materialised."""
    out_a, out_b = [], []
    t_all = torch.as_tensor(tokens)
    for i in range(0, len(t_all), batch):
        logits = model(t_all[i : i + batch])
        pos = torch.as_tensor(positions[i : i + batch])
        rows = torch.arange(len(pos))
        lp = F.log_softmax(logits[rows, pos], -1)  # [b, vocab] for the needed positions only
        out_a.append(lp[rows, torch.as_tensor(ids_a[i : i + batch])].cpu().numpy())
        out_b.append(lp[rows, torch.as_tensor(ids_b[i : i + batch])].cpu().numpy())
        del logits, lp
    return np.concatenate(out_a), np.concatenate(out_b)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["zipper", "divgain", "mix", "random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--prefix", default="code_big")
    p.add_argument("--step", type=int, default=8000)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-docs", type=int, default=2000)
    a = p.parse_args(argv)
    splits, meta = load_dataset(Path("datasets/code_pool.npz"))
    vocab, pad = meta["vocab"], meta["pad_id"]
    ranks = name_ranks(splits["train"]["tokens"], vocab)
    val = splits["val"]["tokens"][: a.n_docs]
    docs = tag_documents(val, vocab, pad, ranks, a.top_k)
    rare_pool = np.array([t for t, r in ranks.items() if r >= a.top_k and r < a.top_k + 3000])
    rng = np.random.default_rng(0)
    pairs, swapped = [], []
    for i, d in enumerate(docs):
        for x, q1, q2 in d["idents"]:
            present = set(val[i].tolist())
            y = int(rng.choice(rare_pool))
            while y in present:
                y = int(rng.choice(rare_pool))
            doc = val[i].copy()
            doc[q1] = y
            pairs.append((i, x, y, q1, q2))
            swapped.append(doc)
    swapped = np.stack(swapped)
    print(f"{len(pairs)} identifier pairs swapped")
    rows = []
    for arm in a.arms:
        for s in a.seeds:
            run = f"{a.prefix}_{arm}_s{s}"
            model, _ = load_checkpoint(Path("checkpoints", run, f"step_{a.step}.pt"))
            model.eval()
            orig = val[[i for i, x, y, q1, q2 in pairs]]
            pos = np.array([q2 - 1 for i, x, y, q1, q2 in pairs])
            xs = np.array([x for i, x, y, q1, q2 in pairs]); ys = np.array([y for i, x, y, q1, q2 in pairs])
            lx, ly = gather_logprobs(model, orig, pos, xs, ys)
            lx_s, ly_s = gather_logprobs(model, swapped, pos, xs, ys)
            rows.append({"arm": arm, "seed": s, "logp_x": lx.mean(), "logp_x_swapped": lx_s.mean(), "x_drop": (lx - lx_s).mean(),
                         "logp_y": ly.mean(), "logp_y_swapped": ly_s.mean(), "y_gain": (ly_s - ly).mean(),
                         "frac_prefers_y_after_swap": float((ly_s > lx_s).mean()), "frac_prefers_x_before": float((lx > ly).mean())})
            print(f"{run}: log p(x) {lx.mean():.2f} -> {lx_s.mean():.2f} after swap (drop {rows[-1]['x_drop']:+.2f}); log p(y) {ly.mean():.2f} -> {ly_s.mean():.2f} (gain {rows[-1]['y_gain']:+.2f}); prefers y after swap {rows[-1]['frac_prefers_y_after_swap']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv("results/swap_control.csv", index=False)
    print("\nper arm (mean over seeds):")
    print(df.groupby("arm")[["x_drop", "y_gain", "frac_prefers_y_after_swap"]].mean().round(3).loc[a.arms].to_string())
    Path("results/swap_control.json").write_text(json.dumps({"step": a.step, "n_pairs": len(pairs), "rows": rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
