"""Skew-reversal trace: where does the "never an answer" prior on held-out names live, and
does its strength track how often the name was seen as the subject in training?

For every pool name n in a run: 64 single-sentence prompts with n as the IO and a random
regular subject; at the answer position, the context bonus of n (logit of n minus the mean
logit of pool names absent from the prompt) decomposed by component (analysis.circuit with
pool_ids). Training counts per name: occurrences as IO, as S, and in total. Output one row
per (run, name) with the MLP-0 / MLP-1 / heads / bias terms and the counts, then
correlations across held-out names and across regular names.

    uv run python -m analysis.skew_trace --runs "zipf*_s*,pool16_s*"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.circuit import _step_of, analyze_checkpoint
from data.generator import OBJECTS, PLACES, VOCAB, load_dataset
from train.train import load_checkpoint


def prompts_for_name(meta: dict, io: int, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    regular = [x for x in meta["regular_name_ids"] if x != io]
    heldout_pairs = {frozenset(p) for p in meta["heldout_pairs"]}
    rows = []
    while len(rows) < n:
        s = int(rng.choice(regular))
        if frozenset((io, s)) in heldout_pairs:
            continue
        a, b = (io, s) if rng.random() < 0.5 else (s, io)
        words = ["[BOS]", "when", a, "and", b, "went", "to", "the", str(rng.choice(PLACES)), ",", s, "gave", "a", str(rng.choice(OBJECTS)), "to", io, "."]
        rows.append(([w if isinstance(w, int) else VOCAB.id[w] for w in words], s))
    tokens = np.array([r[0] for r in rows], dtype=np.int64)
    mask = np.zeros_like(tokens, dtype=bool)
    mask[:, 15] = True
    s_ids = np.full_like(tokens, -1)
    s_ids[:, 15] = [r[1] for r in rows]
    return {"tokens": tokens, "target_mask": mask, "s_ids": s_ids}


def name_counts(splits: dict, meta: dict) -> pd.DataFrame:
    tr = splits["train"]
    b, p = tr["target_mask"].nonzero()
    io_counts = pd.Series(tr["tokens"][b, p]).value_counts()
    s_counts = pd.Series(tr["s_ids"][b, p]).value_counts()
    tot = pd.Series(tr["tokens"].reshape(-1)).value_counts()
    rows = []
    for name in meta["pool_name_ids"]:
        rows.append({"name": name, "count_io": int(io_counts.get(name, 0)), "count_s": int(s_counts.get(name, 0)), "count_total": int(tot.get(name, 0)),
                     "heldout": name in set(meta["heldout_io_name_ids"])})
    return pd.DataFrame(rows)


def trace_run(run: str, results_dir: Path, checkpoints_dir: Path, n_prompts: int) -> pd.DataFrame:
    tc = json.loads((results_dir / run / "train_config.json").read_text())
    splits, meta = load_dataset(Path(tc["dataset"]))
    counts = name_counts(splits, meta)
    ckpt = max((checkpoints_dir / run).glob("step_*.pt"), key=_step_of)
    model, _ = load_checkpoint(ckpt)
    rng = np.random.default_rng(0)
    rows = []
    with torch.no_grad():
        for name in meta["pool_name_ids"]:
            pr = prompts_for_name(meta, int(name), n_prompts, rng)
            out = analyze_checkpoint(model, pr, max_docs=n_prompts, pool_ids=meta["pool_name_ids"])
            c = out["components_ctx"]
            rows.append({"run": run, "name": int(name), "zipf": meta["config"]["name_zipf"], "io_acc": out["io_acc"], "ctx_bonus": out["ctx_bonus"],
                         "mlp0": c.get("mlp_0", 0.0), "mlp1": c.get("mlp_1", 0.0), "heads": c["heads"], "bias": c["bias"], "embed": c["embed"] + c.get("pos_embed", 0.0)})
    return pd.DataFrame(rows).merge(counts, on="name")


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="zipf*_s*,pool16_s*")
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    a = p.parse_args(argv)
    runs = sorted(d.name for d in a.results_dir.iterdir() if d.is_dir() and any(fnmatch.fnmatch(d.name, g) for g in a.runs.split(",")) and (a.checkpoints_dir / d.name).exists())
    df = pd.concat([trace_run(r, a.results_dir, a.checkpoints_dir, a.n_prompts) for r in runs], ignore_index=True)
    df.to_csv(a.results_dir / "skew_trace.csv", index=False)
    print(f"{len(runs)} runs, {len(df)} (run, name) rows; wrote results/skew_trace.csv")
    held, reg = df[df.heldout], df[~df.heldout]
    print("\nheld-out names, by skew (means): count as S, MLP-0 term, MLP-1 term, heads, context bonus, IO accuracy")
    print(held.groupby("zipf")[["count_s", "mlp0", "mlp1", "heads", "ctx_bonus", "io_acc"]].mean().round(2).to_string())
    print("\ncorrelations across held-out (run, name) rows:")
    for x in ("count_s", "count_total"):
        for y in ("mlp0", "mlp1", "ctx_bonus", "io_acc"):
            print(f"  Spearman({x}, {y}) = {held[x].corr(held[y], method='spearman'):+.2f}  (n={len(held)})")
    r1 = reg.count_io.corr(reg.mlp0, method="spearman")
    r2 = reg.count_io.corr(reg.ctx_bonus, method="spearman")
    r3 = reg.count_s.corr(reg.mlp0, method="spearman")
    print(f"\nregular names: Spearman(count_io, mlp0) = {r1:+.2f}, Spearman(count_io, ctx_bonus) = {r2:+.2f}, Spearman(count_s, mlp0) = {r3:+.2f} (n={len(reg)})")
    return df


if __name__ == "__main__":
    main()
