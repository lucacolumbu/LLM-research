"""Clean generalisation for two-name runs: accuracy on single-sentence probes (no
in-context repeat to copy from) for the validation-style and held-out-IO prompt sets, at
the final checkpoint. Merges into results/diversity_pooled.csv when present and reports
the diversity/gain correlations with the clean metric.

    uv run python -m analysis.clean_generalisation --runs "pool*_s*,zipf*_s*,rep*_s*"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.circuit import _step_of
from analysis.patching import Q, build_prompts
from data.generator import load_dataset
from train.train import load_checkpoint


def probe(run: str, n: int) -> dict:
    tc = json.loads(Path("results", run, "train_config.json").read_text())
    _, meta = load_dataset(Path(tc["dataset"]))
    model, _ = load_checkpoint(max(Path("checkpoints", run).glob("step_*.pt"), key=_step_of))
    model.eval()
    out = {"run": run}
    with torch.no_grad():
        for split in ("val", "heldout_io"):
            pr = build_prompts(meta, n, seed=1, split=split)
            if len(pr["tokens"]) == 0:
                out[f"probe_{split}_acc"] = np.nan
                continue
            logits = model(torch.as_tensor(pr["tokens"]))[:, Q]
            ar = torch.arange(len(pr["io"]))
            out[f"probe_{split}_acc"] = (logits.argmax(-1).numpy() == pr["io"]).mean()
            out[f"probe_{split}_ld"] = (logits[ar, torch.as_tensor(pr["io"])] - logits[ar, torch.as_tensor(pr["s"])]).mean().item()
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="pool*_s*,zipf*_s*,rep*_s*")
    p.add_argument("--n", type=int, default=256)
    a = p.parse_args(argv)
    runs = sorted(d.name for d in Path("results").iterdir() if d.is_dir() and any(fnmatch.fnmatch(d.name, g) for g in a.runs.split(",")) and Path("checkpoints", d.name).exists())
    df = pd.DataFrame([probe(r, a.n) for r in runs])
    df.to_csv("results/clean_generalisation.csv", index=False)
    pooled = Path("results/diversity_pooled.csv")
    if pooled.exists():
        d = pd.read_csv(pooled)
        d = d[[c for c in d.columns if not c.startswith("probe_")]].merge(df, on="run", how="left")
        d.to_csv(pooled, index=False)
        print(f"{len(df)} runs probed; merged into {pooled}")
        print(d.groupby(["knob", "value"])[["generalization", "probe_heldout_io_acc", "probe_heldout_io_ld", "probe_val_acc"]].mean().round(3).to_string())
        from scipy.stats import spearmanr
        for target in ("probe_heldout_io_acc", "probe_heldout_io_ld"):
            sub = d[["mean_ncd", "zipper_score", target]].dropna()
            R = sub.rank()
            def partial(x, y, c):
                rx = x - np.polyval(np.polyfit(c, x, 1), c); ry = y - np.polyval(np.polyfit(c, y, 1), c)
                return spearmanr(rx, ry)[0]
            print(f"{target}: Spearman NCD {spearmanr(sub.mean_ncd, sub[target])[0]:+.2f}, gain {spearmanr(sub.zipper_score, sub[target])[0]:+.2f} | partial NCD {partial(R.mean_ncd.values, R[target].values, R.zipper_score.values):+.2f}, gain {partial(R.zipper_score.values, R[target].values, R.mean_ncd.values):+.2f} (n={len(sub)})")
    return df


if __name__ == "__main__":
    main()
