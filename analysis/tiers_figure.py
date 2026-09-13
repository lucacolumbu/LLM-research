"""Formation step against the tier-1 (zipper) and tier-2 (reference-model loss) dataset
scores for every induction run, coloured by knob family. Reads
results/induction_tiers_vs_formation.csv (written by the tier-2 comparison).

    uv run python -m analysis.tiers_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plots import INK, MUTED, SERIES, SURFACE, _style

FAMILY_LABEL = {"repeat_frac": "repeat fraction", "noise": "noise", "vocab_size": "vocabulary size", "n_repeats": "repeat structure"}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/induction_tiers_vs_formation.csv"))
    p.add_argument("--out", type=Path, default=Path("results/tiers_vs_formation.png"))
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
    max_step = int(df["formation_step_acc50"].max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle("Which score predicts induction-head formation? Tier 1 (LZ77) vs tier 2 (reference-model loss)", x=0.02, ha="left", fontsize=12, color=INK)
    for ax, score, title in ((axes[0], "zipper_score", "Tier 1: LZ77 compression gain"), (axes[1], "tier2_gain", "Tier 2: loss gain under a reference model")):
        for i, (fam, g) in enumerate(df.groupby("family")):
            col = SERIES[i % len(SERIES)]
            formed = g["formation_step_acc50"].notna()
            ax.scatter(g[formed][score], g[formed]["formation_step_acc50"], color=col, s=30, zorder=3, label=FAMILY_LABEL.get(fam, fam))
            if (~formed).any():
                ax.scatter(g[~formed][score], [max_step * 1.1] * int((~formed).sum()), facecolor="none", edgecolor=col, s=30, zorder=3)
        f = df.dropna(subset=["formation_step_acc50"])
        rho_f = f[score].corr(f["formation_step_acc50"], method="spearman")
        rho_c = df[score].corr(df["formation_step_acc50"].fillna(max_step * 1.1), method="spearman")
        ax.text(0.98, 0.97, f"Spearman: {rho_f:+.2f} formed runs, {rho_c:+.2f} with never-formed ranked last", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        ax.scatter([], [], facecolor="none", edgecolor=INK, s=30, label="never formed")
        _style(ax, title, "formation step (val accuracy >= 0.5)")
        ax.set_xlabel("dataset score (gain vs shuffled documents)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
