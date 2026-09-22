"""The Stack (Python, 2.5B-token pool): diversity-constrained gain selection vs random.
Four panels from results/stack/trajectories.json and analysis.json: prefix-matching and
induction accuracy over the formation window (three seeds per arm),
validation loss, and the in-context delta on rare identifiers at the final checkpoint.

    uv run python -m analysis.stack_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import INK, MUTED, SERIES, SURFACE, _style

ARMS = [("divgain", "gain + diversity constraint", SERIES[2]), ("random", "random", SERIES[1])]
TOKENS_PER_STEP = 64 * 256


def main() -> None:
    traj = json.loads(Path("results/stack/trajectories.json").read_text())
    analysis = json.loads(Path("results/stack/analysis.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), facecolor=SURFACE)
    fig.suptitle("The Stack, Python: two-quantity selection vs random (4L8H, 491M tokens per arm, three seeds; 1B-token continuation dashed)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    bars = []
    for arm, label, color in ARMS:
        deltas = []
        for s in (0, 1, 2):
            run = f"stack_{arm}_s{s}"
            st, pm, acc = zip(*traj[run]["prefix_matching"])
            kw = {"color": color, "linewidth": 1.3, "alpha": 0.85, "label": label if s == 0 else None}
            axes[0].plot(st, pm, **kw)
            axes[1].plot(st, acc, **kw)
            lst, _, val = zip(*traj[run]["log"])
            axes[2].plot(lst, val, **kw)
            deltas.append(analysis[run]["delta"]["delta_mean"])
        long = f"stack_{arm}_s0_60000"  # seed 0 resumed to 60,000 steps; identical to s0 up to 30,000
        lst, _, val = zip(*traj[long]["log"])
        axes[2].plot(lst, val, color=color, linewidth=1.0, alpha=0.6, linestyle="--")
        bars.append((label, color, np.mean(deltas), np.std(deltas), analysis[long]["delta"]["delta_mean"]))
    for ax in axes[:2]:
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlim(0, 12000)  # the formation window; both measures are flat after this
    _style(axes[0], "Prefix-matching induction score (best head)", "attention to token after previous occurrence")
    _style(axes[1], "Induction accuracy (random repeated sequences)", "accuracy")
    _style(axes[2], "Validation loss", "cross-entropy (nats)")
    axes[2].set_ylim(1.5, 3.0)
    for ax in axes[:3]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        sec = ax.secondary_xaxis("top", functions=(lambda s: s * TOKENS_PER_STEP / 1e6, lambda t: t * 1e6 / TOKENS_PER_STEP))
        sec.set_xlabel("tokens seen (M)", color=MUTED, fontsize=8)
        sec.tick_params(colors=MUTED, labelsize=7)
    ax = axes[3]
    x = np.arange(len(bars))
    ax.bar(x - 0.2, [b[2] for b in bars], 0.38, color=[b[1] for b in bars], yerr=[b[3] for b in bars], capsize=3, label="491M tokens, mean of 3 seeds")
    ax.bar(x + 0.2, [b[4] for b in bars], 0.38, color=[b[1] for b in bars], alpha=0.45, label="1B tokens, seed 0")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0].replace(" + ", "\n+ ") for b in bars], fontsize=8)
    _style(ax, "In-context delta on rare identifiers, final checkpoint", "CE first minus second occurrence (nats)")
    ax.set_xlabel("")
    ax.set_ylim(0, 6.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path("results/stack/stack_trajectories.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
