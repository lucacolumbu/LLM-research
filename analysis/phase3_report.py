"""Phase 3 report: real-text selection by gzip gain vs random. Per subset: prefix-matching
induction score by checkpoint, validation loss by checkpoint, and the LZ77 match-length
histogram of the selected documents (the bridge to the synthetic copy-length results).

    uv run python -m analysis.phase3_report [--prefix text_big]
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import INK, MUTED, SERIES, SURFACE, _style
from analysis.zipper import match_length_stats
from data.generator import load_dataset

ARMS = {"zipper": SERIES[0], "random": SERIES[1]}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="text_big")
    p.add_argument("--pool", type=Path, default=Path("datasets/text_pool_big.npz"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-docs", type=int, default=3000)
    p.add_argument("--out", type=Path, default=Path("results/phase3.png"))
    a = p.parse_args(argv)

    summary: dict = {"arms": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    fig.suptitle("Phase 3: TinyStories subsets selected by gzip gain vs random", x=0.02, ha="left", fontsize=12, color=INK)
    for arm, color in ARMS.items():
        runs = []
        for s in a.seeds:
            run = f"{a.prefix}_{arm}_s{s}"
            rd = Path("results", run)
            if not (rd / "prefix_matching_summary.json").exists():
                continue
            pm = json.loads((rd / "prefix_matching_summary.json").read_text())
            log = [json.loads(line) for line in (rd / "log.jsonl").read_text().splitlines()]
            runs.append({"run": run, "formation_pm": pm["formation_step_prefix_matching"], "formation_acc": pm["formation_step_induction_acc"],
                         "final_pm": pm["final_max_prefix_matching"], "final_ind_acc": pm["final_induction_acc"],
                         "pm_traj": pm["trajectory"], "val_loss": [(r["step"], r["val_loss"]) for r in log], "final_val_loss": log[-1]["val_loss"]})
            steps, pms, _accs = zip(*pm["trajectory"])
            axes[0].plot(steps, pms, color=color, linewidth=1.4, alpha=0.85, label=arm if s == a.seeds[0] else None)
            ls, lv = zip(*runs[-1]["val_loss"])
            axes[1].plot(ls, lv, color=color, linewidth=1.4, alpha=0.85, label=arm if s == a.seeds[0] else None)
        # match-length histogram of the selected documents (seed-0 subset)
        ds = Path("datasets", f"{a.prefix}_{arm}.npz" if arm == "zipper" else f"{a.prefix}_{arm}_s0.npz")
        ml = None
        if ds.exists():
            splits, meta = load_dataset(ds)
            ml = match_length_stats(splits["train"]["tokens"], meta["pad_id"], n_docs=a.n_docs)
            sel = meta.get("selection", {})
            if sel.get("method") != "zipper":  # the stored score is the selection score, not the gzip gain
                from analysis.zipper import Zipper, strip_pad

                rng = np.random.default_rng(0)
                z = Zipper("zlib")
                sample = rng.choice(len(splits["train"]["tokens"]), min(2000, len(splits["train"]["tokens"])), replace=False)
                sel = dict(sel, mean_score_selected=float(np.mean([z.compression_gain(strip_pad(splits["train"]["tokens"][i], meta["pad_id"]), rng) for i in sample])))
        summary["arms"][arm] = {"runs": [{k: v for k, v in r.items() if k not in ("pm_traj", "val_loss")} for r in runs], "match_lengths": ml,
                                "mean_gzip_gain_selected": sel.get("mean_score_selected") if ml else None}
    if a.pool.exists():
        splits, meta = load_dataset(a.pool)
        summary["pool_match_lengths"] = match_length_stats(splits["train"]["tokens"], meta["pad_id"], n_docs=a.n_docs)

    axes[0].axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    _style(axes[0], "Prefix-matching induction score (best head)", "attention to token after previous occurrence")
    _style(axes[1], "Validation loss", "cross-entropy (nats)")
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[2]
    groups = [("pool", summary.get("pool_match_lengths"), MUTED)] + [(arm, summary["arms"][arm]["match_lengths"], c) for arm, c in ARMS.items()]
    groups = [(n, m, c) for n, m, c in groups if m]
    if groups:
        bins = groups[0][1]["bins"]
        labels = [f"{lo}-{hi - 1}" if hi - lo > 1 else f"{lo}" for lo, hi in pairwise(bins)]
        width = 0.8 / len(groups)
        x = np.arange(len(labels))
        for gi, (name, m, c) in enumerate(groups):
            per_doc = np.array(m["hist"]) / max(m["n_docs"], 1)
            ax.bar(x + gi * width - 0.4 + width / 2, per_doc, width, color=c, label=f"{name} (median {m['median_match_len']:.0f}, longest {m['mean_longest_match']:.1f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yscale("log")
    _style(ax, "LZ77 match lengths per document (selected subsets)", "matches per document")
    ax.set_xlabel("match length (tokens)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    Path("results/phase3_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {a.out} and results/phase3_summary.json")
    for arm, d in summary["arms"].items():
        ml = d["match_lengths"] or {}
        print(f"{arm:7s} gzip gain {d['mean_gzip_gain_selected']}: median match {ml.get('median_match_len')}, mean longest {ml.get('mean_longest_match')}, "
              f"frac tokens in matches>=8 {ml.get('frac_tokens_in_matches_ge8')}")
        for r in d["runs"]:
            print(f"   {r['run']}: formation pm {r['formation_pm']} acc {r['formation_acc']}; final pm {r['final_pm']:.2f} ind-acc {r['final_ind_acc']:.2f} val loss {r['final_val_loss']:.3f}")
    return summary


if __name__ == "__main__":
    main()
