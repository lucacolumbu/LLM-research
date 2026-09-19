"""Four-arm comparison on code: gain-only, random, mixed, diversity-constrained.
Prefix-matching by checkpoint, validation loss by checkpoint, and loss on repeated
low-frequency identifiers at the final checkpoint.

    uv run python -m analysis.four_arms_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import INK, MUTED, SERIES, SURFACE, _style

ARMS = [("zipper", "gain only", SERIES[0]), ("divgain", "gain + diversity constraint", SERIES[2]), ("mix", "half gain, half random", SERIES[3]), ("random", "random", SERIES[1])]


def main() -> None:
    idl = json.loads(Path("results/identifier_loss.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    fig.suptitle("Python source: one compressor quantity vs two (three seeds per arm, 8,000 steps)", x=0.02, ha="left", fontsize=12, color=INK)
    bars = []
    for arm, label, color in ARMS:
        ids, oth = [], []
        for s in (0, 1, 2):
            r = f"code_big_{arm}_s{s}"
            pm = json.loads(Path("results", r, "prefix_matching_summary.json").read_text())
            log = [json.loads(line) for line in Path("results", r, "log.jsonl").read_text().splitlines()]
            st, pms, _ = zip(*pm["trajectory"])
            axes[0].plot(st, pms, color=color, linewidth=1.3, alpha=0.85, label=label if s == 0 else None)
            axes[1].plot([x["step"] for x in log], [x["val_loss"] for x in log], color=color, linewidth=1.3, alpha=0.85, label=label if s == 0 else None)
            ids.append(idl[r]["repeat_ident"]["ce"]); oth.append(idl[r]["other"]["ce"])
        bars.append((label, color, np.mean(ids), np.std(ids), np.mean(oth)))
    _style(axes[0], "Prefix-matching induction score (best head)", "attention to token after previous occurrence")
    _style(axes[1], "Validation loss", "cross-entropy (nats)")
    axes[1].set_ylim(1.8, 3.2)
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[2]
    x = np.arange(len(bars))
    ax.bar(x - 0.2, [b[2] for b in bars], 0.38, color=[b[1] for b in bars], yerr=[b[3] for b in bars], capsize=3, label="repeated low-frequency identifiers")
    ax.bar(x + 0.2, [b[4] for b in bars], 0.38, color=[b[1] for b in bars], alpha=0.45, label="other tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0].replace(", ", ",\n") for b in bars], fontsize=8)
    _style(ax, "Loss where copying should pay, final checkpoint", "cross-entropy (nats)")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("results/phase3_code_four_arms.png", dpi=150, facecolor=SURFACE)
    print("wrote results/phase3_code_four_arms.png")


if __name__ == "__main__":
    main()
