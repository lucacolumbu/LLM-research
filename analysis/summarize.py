"""Aggregate a sweep: one table row per knob value (mean and std over seeds) and a figure.

    uv run python -m analysis.summarize --knob name_pool_size --runs "pool*_s*"
reads results/results.csv plus results/<run>/circuit_summary.json and circuit.jsonl, and
writes results/summary_<knob>.csv and results/summary_<knob>.png.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.plots import GRID, INK, MUTED, SURFACE, _style

# knobs added after some runs were generated; older condition JSON lacks them
KNOB_DEFAULTS = {"n_repeats": 1, "repeat_frac_min": 0.0, "heldout_io_leak": 0.0, "noise": 0.0, "vocab_size": 100}
BLUE_RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordered knob values


def load_runs(results_dir: Path, pattern: str, knob: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "results.csv")
    patterns = pattern.split(",")
    df = df[[any(fnmatch.fnmatch(r, p) for p in patterns) for r in df["run"]]].drop_duplicates("run", keep="last").copy()
    df[knob] = [json.loads(c).get(knob, KNOB_DEFAULTS.get(knob, np.nan)) for c in df["condition"]]
    extra = []
    for run in df["run"]:
        path = results_dir / run / "circuit_summary.json"
        rec: dict[str, object] = {"run": run}
        if path.exists():
            s = json.loads(path.read_text())
            gen = s["generalisation_split"]
            comp = s["final_components_ld"].get(gen, {})
            rec.update(
                formation_step_attn_s=s.get("formation_step_attn_s"),
                formation_step_attn_io=s.get("formation_step_attn_io"),
                formation_step_attn_lag=s.get("formation_step_attn_lag"),
                dominant_lag=(max((m, lag) for row_m, row_l in zip(s["final_best_lag_mass"], s["final_best_lag"]) for m, lag in zip(row_m, row_l))[1] if s.get("final_best_lag_mass") else None),
                crossing_step=s.get("crossing_step_logit_diff"),
                sustained_crossing_step=s.get("sustained_crossing_step_logit_diff"),
                min_ld=(s.get("min_logit_diff") or [np.nan])[0],
                heads_ld=comp.get("heads"),
                mlp0_ld=comp.get("mlp_0"),
                mlp1_ld=comp.get("mlp_1"),
                prior_ld=comp.get("bias"),
                top_heads=",".join(f"L{layer}H{head}" for layer, head in s["name_mover_candidates"]),
            )
        lpath = results_dir / run / "log.jsonl"
        if lpath.exists():
            log = [json.loads(line) for line in lpath.read_text().splitlines()]
            rec["formation_step_acc90"] = next((r["step"] for r in log if r.get("val_io_acc", 0) >= 0.9), None)
            rec["formation_step_acc50"] = next((r["step"] for r in log if r.get("val_io_acc", 0) >= 0.5), None)
            accs = [r.get("val_io_acc", 0) for r in log]
            rec["max_acc_gain_per_eval"] = max((b - a for a, b in pairwise(accs)), default=None)
        ppath = results_dir / run / "patching_summary.json"
        if ppath.exists():
            ps = json.loads(ppath.read_text())
            fv = ps.get("final_val", {})
            fh = ps.get("final_heldout_io", {})
            rec.update(
                circuit_heads=",".join(ps["circuit_heads"]),
                formation_step_patching=ps.get("formation_step_patching"),
                faithfulness_p=ps.get("faithfulness"),
                sharpness_p=ps.get("sharpness"),
                circuit_recovery=fv.get("circuit_recovery"),
                ablate_acc_val=fv.get("ablate_circuit_mean_acc"),
                ctx_bonus_val=fv.get("ctx_bonus"),
                ablate_acc_heldout=fh.get("ablate_circuit_mean_acc"),
                ctx_bonus_heldout=fh.get("ctx_bonus"),
                ctx_mlp1_val=(ps.get("final_val_components_ctx") or {}).get("mlp_1"),
                ctx_heads_val=(ps.get("final_val_components_ctx") or {}).get("heads"),
            )
        extra.append(rec)
    return df.merge(pd.DataFrame(extra), on="run", how="left")


def trajectory(results_dir: Path, run: str, split: str, key: str) -> tuple[list[int], list[float]]:
    path = results_dir / run / "circuit.jsonl"
    if not path.exists():
        return [], []
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    return [r["step"] for r in recs], [r[split][key] if r.get(split) else np.nan for r in recs]


METRICS = [
    "zipper_score", "val_io_acc", "generalization", "generalization_pairs",
    "formation_step_attn_s", "formation_step_attn_io", "formation_step_attn_lag", "dominant_lag", "formation_step_acc90", "formation_step_acc50", "max_acc_gain_per_eval", "crossing_step", "sustained_crossing_step", "min_ld",
    "heads_ld", "mlp0_ld", "mlp1_ld", "prior_ld",
    "formation_step_patching", "faithfulness_p", "sharpness_p", "circuit_recovery", "ablate_acc_val",
    "ablate_acc_heldout", "ctx_bonus_val", "ctx_bonus_heldout", "ctx_mlp1_val", "ctx_heads_val",
]


def summarize(df: pd.DataFrame, knob: str) -> pd.DataFrame:
    g = df.groupby(knob)
    out = pd.DataFrame({"n_seeds": g.size()})
    for m in METRICS:
        if m in df:
            vals = pd.to_numeric(df[m], errors="coerce")
            out[f"{m}_mean"] = vals.groupby(df[knob]).mean()
            out[f"{m}_std"] = vals.groupby(df[knob]).std()
    out["crossing_frac"] = g["crossing_step"].apply(lambda s: s.notna().mean()) if "crossing_step" in df else np.nan
    return out.reset_index()


def plot(df: pd.DataFrame, knob: str, results_dir: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    values = sorted(df[knob].unique())
    colors = [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(len(values) - 1, 1))] for i in range(len(values))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURFACE)
    fig.suptitle(f"sweep over {knob}", x=0.02, ha="left", fontsize=12, color=INK)

    ax = axes[0]
    for v, c in zip(values, colors):
        runs = df[df[knob] == v]["run"]
        series = [trajectory(results_dir, r, "heldout_io", "logit_diff") for r in runs]
        series = [s for s in series if s[0]]
        if not series:
            continue
        steps = series[0][0]
        for _, y in series:
            ax.plot(steps, y, color=c, linewidth=0.7, alpha=0.5)
        ax.plot(steps, np.nanmean([y for _, y in series], axis=0), color=c, linewidth=1.8, label=f"{knob}={v:g}")
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    _style(ax, "Held-out IO logit difference (thin: seeds, thick: mean)", "IO minus S logits")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[1]
    for v, c in zip(values, colors):
        sub = df[df[knob] == v]
        ax.scatter([v] * len(sub), sub["generalization"], color=c, s=28, zorder=3)
        ax.scatter([v] * len(sub), sub["generalization_pairs"], facecolor="none", edgecolor=c, s=28, zorder=3)
    ax.scatter([], [], color=INK, s=28, label="held-out IO names")
    ax.scatter([], [], facecolor="none", edgecolor=INK, s=28, label="held-out pairs")
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Final IO accuracy per seed", "accuracy")
    ax.set_xlabel(knob, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[2]
    for v, c in zip(values, colors):
        sub = df[df[knob] == v]
        ax.scatter([v] * len(sub), pd.to_numeric(sub["formation_step_attn_s"], errors="coerce"), color=c, s=28, zorder=3)
        ax.scatter([v] * len(sub), pd.to_numeric(sub["crossing_step"], errors="coerce"), facecolor="none", edgecolor=c, s=28, zorder=3)
    ax.scatter([], [], color=INK, s=28, label="S-attention formation step")
    ax.scatter([], [], facecolor="none", edgecolor=INK, s=28, label="held-out IO zero crossing")
    _style(ax, "Formation and crossing steps per seed (missing = never)", "training step")
    ax.set_xlabel(knob, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    for ax in axes[1:]:
        ax.set_xticks(values)
        ax.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knob", required=True)
    p.add_argument("--runs", default="*_s*", help="comma-separated globs over run names")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--correlate", default="", help="x:y column pair to correlate across runs, e.g. zipper_score:formation_step_acc90")
    a = p.parse_args(argv)

    df = load_runs(a.results_dir, a.runs, a.knob)
    if df.empty:
        raise SystemExit(f"no runs match {a.runs!r}")
    table = summarize(df, a.knob)
    if a.correlate:
        x, y = a.correlate.split(":")
        sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        print(f"Spearman({x}, {y}) over {len(sub)} runs: {sub[x].corr(sub[y], method='spearman'):+.3f}; "
              f"Pearson: {sub[x].corr(sub[y]):+.3f}")
    table.to_csv(a.results_dir / f"summary_{a.knob}.csv", index=False)
    df.to_csv(a.results_dir / f"runs_{a.knob}.csv", index=False)
    plot(df, a.knob, a.results_dir, a.results_dir / f"summary_{a.knob}.png")
    with pd.option_context("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3g}".format):
        print(table)
    print(f"wrote {a.results_dir / f'summary_{a.knob}.csv'} and .png")
    return table


if __name__ == "__main__":
    main()
