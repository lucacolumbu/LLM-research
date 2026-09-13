"""Tier-1 test figure: zipper score against induction formation step, plus the accuracy
trajectories that show whether formation is a snap or a crawl.

    uv run python -m analysis.zipper_figure --runs "ind_rep*_s*" --knob repeat_frac
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.plots import INK, MUTED, SURFACE, _style
from analysis.summarize import BLUE_RAMP, load_runs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="ind_rep*_s*")
    p.add_argument("--knob", default="repeat_frac")
    p.add_argument("--formation", default="formation_step_acc50")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/zipper_vs_formation.png"))
    a = p.parse_args(argv)
    df = load_runs(a.results_dir, a.runs, a.knob)
    values = sorted(df[a.knob].unique())
    colors = {v: BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(len(values) - 1, 1))] for i, v in enumerate(values)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    fig.suptitle("Tier 1: does LZ77 compression gain predict induction-head formation?", x=0.02, ha="left", fontsize=12, color=INK)

    ax = axes[0]
    max_step = max(json.loads(Path(a.results_dir, r, "log.jsonl").read_text().splitlines()[-1])["step"] for r in df["run"])
    formed = df[a.formation].notna()
    for v in values:
        sub = df[df[a.knob] == v]
        f = sub[formed[sub.index]]
        n = sub[~formed[sub.index]]
        ax.scatter(f["zipper_score"], f[a.formation], color=colors[v], s=34, zorder=3, label=f"{a.knob}={v:g}")
        if len(n):
            ax.scatter(n["zipper_score"], [max_step * 1.08] * len(n), facecolor="none", edgecolor=colors[v], s=34, zorder=3)
    ax.scatter([], [], facecolor="none", edgecolor=INK, s=34, label=f"never formed within {max_step} steps")
    sub = df[formed][["zipper_score", a.formation]].astype(float)
    rho = sub["zipper_score"].corr(sub[a.formation], method="spearman") if len(sub) > 2 else float("nan")
    ax.text(0.98, 0.97, f"Spearman rho = {rho:+.2f} (n = {len(sub)} formed runs)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    _style(ax, "Formation step vs per-document zipper gain", "formation step (val accuracy >= 0.5)")
    ax.set_xlabel("mean per-document LZ77 compression gain", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")

    ax = axes[1]
    for v in values:
        for i, r in enumerate(df[df[a.knob] == v]["run"]):
            log = [json.loads(line) for line in Path(a.results_dir, r, "log.jsonl").read_text().splitlines()]
            ax.plot([x["step"] for x in log], [x["val_io_acc"] for x in log], color=colors[v], linewidth=1.2, alpha=0.8, label=f"{a.knob}={v:g}" if i == 0 else None)
    _style(ax, "Validation accuracy on copied tokens (3 seeds each)", "accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}; rho={rho:+.3f}")


if __name__ == "__main__":
    main()
