"""Induction-head formation on any model via the prefix-matching score of the
induction-heads paper: feed [BOS] + a random token sequence of length L repeated twice;
for query positions in the second copy, the score of a head is its mean attention to the
position right after the previous occurrence of the current token. Also reports the
behavioural induction accuracy (argmax next-token prediction on the second copy).

    uv run python -m analysis.prefix_matching --run text_zipper_s0
writes results/<run>/prefix_matching.jsonl and prefix_matching_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.circuit import _step_of
from data.generator import load_dataset
from train.train import load_checkpoint


def repeated_sequences(n: int, length: int, lo: int, hi: int, bos_id: int, seed: int = 0, token_pool: np.ndarray | None = None) -> np.ndarray:
    """Random sequences drawn from `token_pool` if given (e.g. the most frequent real-text
    tokens, whose embeddings are trained), else uniformly from [lo, hi)."""
    rng = np.random.default_rng(seed)
    u = rng.choice(token_pool, size=(n, length)) if token_pool is not None else rng.integers(lo, hi, size=(n, length))
    return np.concatenate([np.full((n, 1), bos_id), u, u], axis=1).astype(np.int64)


@torch.no_grad()
def score_checkpoint(model, seqs: np.ndarray, length: int) -> dict[str, Any]:
    model.eval()
    tok = torch.as_tensor(seqs).to(model.cfg.device)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    names = {f"blocks.{layer}.attn.hook_pattern" for layer in range(L)}
    logits, cache = model.run_with_cache(tok, names_filter=lambda n: n in names)
    q = torch.arange(length + 1, 2 * length, device=tok.device)  # second copy, all but its last token
    k = q - length + 1  # position after the previous occurrence of the current token
    scores = np.zeros((L, H))
    for layer in range(L):
        pat = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, q, :]  # [n, H, |q|, ctx]
        scores[layer] = pat.gather(3, k[None, None, :, None].expand(pat.shape[0], H, -1, 1)).squeeze(-1).mean((0, 2)).cpu().numpy()
    pred = logits[:, q].argmax(-1)
    acc = (pred == tok[:, q + 1]).float().mean().item()
    return {"prefix_matching": scores.tolist(), "max_prefix_matching": float(scores.max()), "induction_acc": acc}


def analyze_run(
    run: str, results_dir: Path = Path("results"), checkpoints_dir: Path = Path("checkpoints"),
    n: int = 128, length: int = 32, threshold: float = 0.5, top_k: int = 512,
) -> dict[str, Any]:
    run_dir = results_dir / run
    tc = json.loads((run_dir / "train_config.json").read_text())
    _, meta = load_dataset(Path(tc["dataset"]))
    lo = len([t for t in meta["vocab"][:4] if t.startswith("[")])  # skip special tokens
    pool = None
    if meta.get("task") == "text":
        # vocab is frequency-ordered for text corpora: probe with the top `top_k` trained tokens
        pool = np.arange(lo, min(lo + top_k, len(meta["vocab"])))
    seqs = repeated_sequences(n, length, lo, len(meta["vocab"]), meta["bos_id"], token_pool=pool)
    records = []
    for path in sorted((checkpoints_dir / run).glob("step_*.pt"), key=_step_of):
        model, ckpt = load_checkpoint(path)
        records.append({"step": ckpt["step"], **score_checkpoint(model, seqs, length)})
    with (run_dir / "prefix_matching.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {
        "formation_step_prefix_matching": next((r["step"] for r in records if r["step"] > 0 and r["max_prefix_matching"] >= threshold), None),
        "formation_step_induction_acc": next((r["step"] for r in records if r["step"] > 0 and r["induction_acc"] >= threshold), None),
        "final_max_prefix_matching": records[-1]["max_prefix_matching"],
        "final_induction_acc": records[-1]["induction_acc"],
        "trajectory": [(r["step"], round(r["max_prefix_matching"], 3), round(r["induction_acc"], 3)) for r in records],
    }
    (run_dir / "prefix_matching_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--top-k", type=int, default=512, help="text corpora: probe with the k most frequent tokens")
    a = p.parse_args(argv)
    s = analyze_run(a.run, n=a.n, length=a.length, top_k=a.top_k)
    print(json.dumps({k: v for k, v in s.items() if k != "trajectory"}, indent=2))
    return s


if __name__ == "__main__":
    main()
