"""Static figures for circuit trajectories. One measure per axis; fixed categorical colours."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # blue, orange, aqua, yellow, magenta
INK, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SPLIT_COLOR = {"val": SERIES[0], "heldout_pairs": SERIES[2], "heldout_io": SERIES[1]}
SPLIT_LABEL = {"val": "validation", "heldout_pairs": "held-out pairs", "heldout_io": "held-out IO names"}


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel("training step", color=MUTED, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_trajectory(records: list[dict[str, Any]], summary: dict[str, Any], run: str, out: Path) -> Path:
    steps = [r["step"] for r in records]
    gen = summary["generalisation_split"]
    top = [tuple(h) for h in summary["name_mover_candidates"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    fig.suptitle(run, x=0.02, ha="left", fontsize=12, color=INK)

    ax = axes[0, 0]
    for split in ("val", "heldout_pairs", "heldout_io"):
        if records[-1].get(split):
            ax.plot(steps, [r[split]["io_acc"] for r in records], color=SPLIT_COLOR[split],
                    linewidth=1.6, label=SPLIT_LABEL[split])
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(steps[0], 0.51, "chance (two candidates)", color=MUTED, fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Indirect-object accuracy", "accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[0, 1]
    for split in ("val", "heldout_pairs", "heldout_io"):
        if records[-1].get(split):
            ax.plot(steps, [r[split]["logit_diff"] for r in records], color=SPLIT_COLOR[split],
                    linewidth=1.6, label=SPLIT_LABEL[split])
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    _style(ax, "Logit difference, IO minus S", "logits")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[1, 0]
    n_layers = len(records[-1][gen]["attn_io"])
    n_heads = len(records[-1][gen]["attn_io"][0])
    for layer in range(n_layers):
        for head in range(n_heads):
            if (layer, head) in top:
                color = SERIES[top.index((layer, head))]
                ax.plot(steps, [r[gen]["attn_s"][layer][head] for r in records], color=color,
                        linewidth=1.6, label=f"L{layer}H{head} to S")
                ax.plot(steps, [r[gen]["attn_io"][layer][head] for r in records], color=color,
                        linewidth=1.6, linestyle=":", label=f"L{layer}H{head} to IO")
            else:
                ax.plot(steps, [r[gen]["attn_io"][layer][head] for r in records], color=GRID, linewidth=1.0)
    ax.plot([], [], color=GRID, linewidth=1.0, label="other heads, to IO")
    ax.axhline(summary["attn_threshold"], color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, f"Attention of top-attribution heads, {SPLIT_LABEL[gen]}", "attention mass")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)

    ax = axes[1, 1]
    comp = [r[gen]["components_ld"] for r in records]
    cand = [sum(r[gen]["dla_ld"][layer][head] for layer, head in top) for r in records]
    heads_total = [c["heads"] for c in comp]
    other = [h - c for h, c in zip(heads_total, cand)]
    mlps = [sum(v for k, v in c.items() if k.startswith("mlp_")) for c in comp]
    embeds = [c["embed"] + c.get("pos_embed", 0.0) + sum(v for k, v in c.items() if k.startswith("attn_bias_")) for c in comp]
    bias = [c["bias"] for c in comp]
    for series, label, color in (
        (cand, "top-attribution heads", SERIES[0]),
        (other, "other heads", SERIES[1]),
        (mlps, "MLPs", SERIES[2]),
        (embeds, "embeddings + attn biases", SERIES[3]),
        (bias, "prior: LN bias + unembed bias", SERIES[4]),
    ):
        ax.plot(steps, series, color=color, linewidth=1.6, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    _style(ax, f"Direct logit attribution to IO minus S, {SPLIT_LABEL[gen]}", "logits")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
