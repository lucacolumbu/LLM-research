"""Phase 2 experiment driver: heterogeneous pool -> subset selection by method -> train ->
compare induction formation. Resumable; each stage is skipped when its output exists.

    uv run python -m selection.experiment --n 20000 --seeds 0 1 2 --jobs 3

Stages:
1. pool          datasets/pool_het.npz: 60k docs, per-document repeat fraction ~ U(0.1, 0.98)
2. random arm    selection per seed, train (attention-only rotary, 6000 steps)
3. reference     the seed-0 random arm's final checkpoint (a model trained on a random
                 subset) scores the pool for the refloss arm
4. other arms    zipper / refloss / oracle selection (deterministic), train per seed
5. summary       results/selection_summary.csv: formation step per arm and seed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

TRAIN_ARGS = ["--positional-embedding-type", "rotary", "--attn-only"]


def sh(cmd: list[str], log: Path, threads: int) -> None:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    with log.open("a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {' '.join(cmd)}\n")
        f.flush()
        subprocess.run([sys.executable, "-m", *cmd], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def train_run(run: str, dataset: Path, seed: int, a: argparse.Namespace) -> None:
    run_dir = Path("results") / run
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "sweep.log"
    if not (Path("checkpoints") / run / f"step_{a.steps}.pt").exists():
        sh(["train.train", "--dataset", str(dataset), "--run", run, "--seed", str(seed), "--steps", str(a.steps),
            "--ckpt-every", str(a.ckpt_every), "--eval-every", str(a.ckpt_every), *TRAIN_ARGS], log, a.threads)
    sh(["analysis.zipper", "--dataset", str(dataset), "--update-results"], log, a.threads)
    if not (run_dir / "circuit_summary.json").exists():
        sh(["analysis.circuit", "--run", run], log, a.threads)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--pool-size", type=int, default=60000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=["zipper", "refloss", "oracle"])
    a = p.parse_args(argv)
    Path("results").mkdir(exist_ok=True)
    log = Path("results/selection_experiment.log")

    pool = Path("datasets/pool_het.npz")
    if not pool.exists():
        sh(["data.induction", "--out", str(pool), "--n-train", str(a.pool_size), "--n-val", "2000",
            "--repeat-frac", "0.98", "--repeat-frac-min", "0.1", "--seed", "0"], log, a.threads)

    def arm(method: str, seed: int) -> tuple[str, Path]:
        ds = Path("datasets") / f"sel_{method}_s{seed}.npz"
        if not ds.exists():
            cmd = ["selection.select_by_score", "--pool", str(pool), "--method", method, "--n", str(a.n),
                   "--seed", str(seed), "--out", str(ds)]
            if method == "refloss":
                cmd += ["--reference", f"checkpoints/sel_random_s0/step_{a.steps}.pt"]
            sh(cmd, log, a.threads)
        return f"sel_{method}_s{seed}", ds

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda s: train_run(*arm("random", s), s, a), a.seeds))
        jobs = [(m, s) for m in a.methods for s in a.seeds]
        list(ex.map(lambda ms: train_run(*arm(ms[0], ms[1]), ms[1], a), jobs))

    rows = []
    for m in ["random", *a.methods]:
        for s in a.seeds:
            run = f"sel_{m}_s{s}"
            logp = Path("results") / run / "log.jsonl"
            recs = [json.loads(line) for line in logp.read_text().splitlines()]
            meta_sel = json.loads(Path("results", run, "train_config.json").read_text())
            sel = None
            dsp = Path("datasets") / f"sel_{m}_s{s}.npz"
            import numpy as np
            with np.load(dsp) as f:
                sel = json.loads(str(f["meta"]))["selection"]
            rows.append({
                "method": m, "seed": s,
                "formation_step_acc50": next((r["step"] for r in recs if r["val_io_acc"] >= 0.5), None),
                "final_val_acc": recs[-1]["val_io_acc"],
                "mean_doc_frac_selected": sel["mean_doc_frac_selected"],
                "score_vs_hidden_spearman": sel["spearman_score_vs_hidden_frac"],
                "dataset": meta_sel["dataset"],
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/selection_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
        print(df.groupby("method")[["formation_step_acc50", "final_val_acc", "mean_doc_frac_selected"]].agg(["mean", "std"]))
    return df


if __name__ == "__main__":
    main()
