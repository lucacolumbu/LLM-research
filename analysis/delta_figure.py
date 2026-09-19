"""Trajectory of the in-context delta (CE at first minus second occurrence of rare
identifiers) by checkpoint for each arm, seeds as thin lines, mean thick.

    uv run python -m analysis.delta_figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plots import INK, SERIES, SURFACE, _style

ARMS = [("zipper", "gain only", SERIES[0]), ("divgain", "gain + diversity constraint", SERIES[2]), ("mix", "half gain, half random", SERIES[3]), ("random", "random", SERIES[1])]


def main() -> None:
    d = pd.read_csv("results/incontext_delta_trajectory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    fig.suptitle("Python source: in-context delta on rare identifiers (CE at first minus second occurrence)", x=0.02, ha="left", fontsize=12, color=INK)
    for arm, label, color in ARMS:
        g = d[d.arm == arm]
        for s, gs in g.groupby("seed"):
            axes[0].plot(gs.step, gs.delta_mean, color=color, linewidth=0.8, alpha=0.5)
            axes[1].plot(gs.step, gs.ce1, color=color, linewidth=0.8, alpha=0.5)
        m = g.groupby("step")[["delta_mean", "ce1"]].mean()
        axes[0].plot(m.index, m.delta_mean, color=color, linewidth=2, label=label)
        axes[1].plot(m.index, m.ce1, color=color, linewidth=2, label=label)
    _style(axes[0], "Mean in-context delta by checkpoint", "nats (positive = better after seeing the name)")
    _style(axes[1], "CE at the first occurrence (distribution quality)", "nats")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("results/incontext_delta_trajectory.png", dpi=150, facecolor=SURFACE)
    print("wrote results/incontext_delta_trajectory.png")


if __name__ == "__main__":
    main()
