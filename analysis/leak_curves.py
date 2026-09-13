"""Sample-efficiency figure for the soft-leak condition: held-out IO accuracy and logit
difference against the number of leaked (held-out-name-as-IO) training examples seen.

    uv run python -m analysis.leak_curves --runs fine_leak0.045_s0 fine_leak0.045_s1 fine_leak0.045_s2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import INK, MUTED, SERIES, SURFACE, _style
from data.generator import load_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("results/leak_sample_efficiency.png"))
    a = p.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    fig.suptitle("Overwriting the held-out-IO prior: examples needed", x=0.02, ha="left", fontsize=12, color=INK)
    first_half = []
    for i, run in enumerate(a.runs):
        tc = json.loads(Path("results", run, "train_config.json").read_text())
        _, meta = load_dataset(Path(tc["dataset"]))
        per_step = meta["stats"]["train"]["frac_heldout_io_sentences"] * tc["batch_size"] * meta["stats"]["train"]["sentences_per_doc"]
        log = [json.loads(line) for line in Path("results", run, "log.jsonl").read_text().splitlines()]
        x = np.array([r["step"] * per_step for r in log])
        acc = np.array([r["heldout_io_io_acc"] for r in log])
        ld = np.array([r["heldout_io_io_logit_diff"] for r in log])
        color = SERIES[i % len(SERIES)]
        axes[0].plot(x, acc, color=color, linewidth=1.6, marker="o", markersize=3, label=run)
        axes[1].plot(x, ld, color=color, linewidth=1.6, marker="o", markersize=3, label=run)
        half = next((xx for xx, aa in zip(x, acc) if aa >= 0.5), None)
        first_half.append(half)
        print(f"{run}: ~{per_step:.2f} leaked examples/step; first checkpoint with acc >= 0.5 at ~{half:.0f} examples"
              if half is not None else f"{run}: never reached 0.5")
    axes[0].axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    axes[0].set_ylim(-0.02, 1.02)
    _style(axes[0], "Held-out IO accuracy", "accuracy")
    axes[1].axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    _style(axes[1], "Held-out IO logit difference", "IO minus S logits")
    for ax in axes:
        ax.set_xlabel("leaked examples seen (held-out name as IO)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
