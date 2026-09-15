"""Corpus diversity (mean pairwise NCD) per run, merged with the run's outcomes, plus
Spearman correlations of diversity and per-document gain with generalisation and formation.

    uv run python -m analysis.diversity --knob name_pool_size --runs "pool*_s*"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.summarize import load_runs
from analysis.zipper import corpus_diversity
from data.generator import load_dataset


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--knob", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--n-pairs", type=int, default=3000)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    a = p.parse_args(argv)
    df = load_runs(a.results_dir, a.runs, a.knob)
    rows = []
    for run in df["run"]:
        tc = json.loads((a.results_dir / run / "train_config.json").read_text())
        splits, meta = load_dataset(Path(tc["dataset"]))
        rows.append({"run": run, **corpus_diversity(splits["train"]["tokens"], meta["pad_id"], a.n_pairs)})
    df = df.merge(pd.DataFrame(rows), on="run")
    out = a.results_dir / f"diversity_{a.knob}.csv"
    df.to_csv(out, index=False)
    cols = [c for c in ("generalization", "generalization_pairs", "formation_step_attn_s", "formation_step_patching", "sustained_crossing_step", "val_io_acc") if c in df]
    print(f"{len(df)} runs; wrote {out}")
    print(df.groupby(a.knob)[["mean_ncd", "zipper_score", *cols]].mean().round(3).to_string())
    print("\nSpearman across runs:")
    for score in ("mean_ncd", "zipper_score"):
        for c in cols:
            sub = df[[score, c]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) > 2:
                rho = sub[score].corr(sub[c], method="spearman")
                print(f"  rho({score}, {c}) = {rho:+.3f}  (n={len(sub)})")
    return df


if __name__ == "__main__":
    main()
