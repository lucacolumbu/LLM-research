"""Run a one-knob sweep: for each (value, seed) generate the dataset, train, compute the
tier-1 zipper score and the head-level circuit trajectory. Resumable: finished stages
are skipped.

    uv run python -m train.sweep --knob name_pool_size --values 4 8 16 32 64 --seeds 0 1 2
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

SHORT = {
    "name_pool_size": "pool",
    "heldout_io_leak": "leak",
    "repetition_rate": "rep",
    "target_noise": "noise",
    "distractor_ratio": "dis",
    "name_zipf": "zipf",
    "n_heldout_io_names": "hold",
    "heldout_pair_frac": "pairs",
    "repeat_frac": "ind_rep",
    "vocab_size": "ind_vocab",
    "noise": "ind_noise",
}


def run_name(knob: str, value: float, seed: int, prefix: str = "") -> str:
    return f"{prefix}{SHORT.get(knob, knob)}{value:g}_s{seed}"


def _cli(overrides: dict[str, object]) -> list[str]:
    out: list[str] = []
    for k, v in overrides.items():
        flag = k.replace("_", "-")
        if isinstance(v, bool):
            out.append(f"--{flag}" if v else f"--no-{flag}")
        else:
            out += [f"--{flag}", str(v)]
    return out


def pipeline(run: str, knob: str, value: float, seed: int, a: argparse.Namespace) -> str:
    dataset = Path("datasets") / f"{run}.npz"
    run_dir = Path("results") / run
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "sweep.log").open("a")
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    def stage(name: str, cmd: list[str], done: bool) -> None:
        if done:
            log.write(f"[{time.strftime('%H:%M:%S')}] skip {name}\n")
            return
        log.write(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(cmd)}\n")
        log.flush()
        subprocess.run([sys.executable, "-m", *cmd], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    base = dict(json.loads(a.base), seed=seed)
    base[knob] = value
    stage("generate", [a.generator, "--out", str(dataset), *_cli(base)], dataset.exists())
    final_ckpt = Path("checkpoints") / run / f"step_{a.steps}.pt"
    train_args = dict(json.loads(a.train_args), seed=seed, steps=a.steps, ckpt_every=a.ckpt_every, eval_every=a.eval_every)
    stage("train", ["train.train", "--dataset", str(dataset), "--run", run, *_cli(train_args)], final_ckpt.exists())
    stage("zipper", ["analysis.zipper", "--dataset", str(dataset), "--update-results"], False)
    stage("circuit", ["analysis.circuit", "--run", run], (run_dir / "circuit_summary.json").exists())
    log.close()
    return run


def main(argv: list[str] | None = None) -> list[str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knob", required=True, help="DataConfig field to sweep")
    p.add_argument("--values", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--base", default="{}", help="JSON of DataConfig overrides shared by all runs")
    p.add_argument("--train-args", default="{}", help="JSON of TrainConfig overrides")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--threads", type=int, default=2, help="torch threads per job")
    p.add_argument("--prefix", default="")
    p.add_argument("--generator", default="data.generator", help="generator module, e.g. data.induction")
    a = p.parse_args(argv)

    int_knobs = {"name_pool_size", "n_heldout_io_names", "ctx_len", "n_train", "n_val", "vocab_size"}
    jobs = []
    for v in a.values:
        value = int(v) if a.knob in int_knobs else v
        for seed in a.seeds:
            jobs.append((run_name(a.knob, value, seed, a.prefix), a.knob, value, seed))
    print(f"{len(jobs)} runs, {a.jobs} in parallel: {[j[0] for j in jobs]}", flush=True)

    t0 = time.time()
    done: list[str] = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futures = {ex.submit(pipeline, *j, a): j[0] for j in jobs}
        for fut, name in futures.items():
            try:
                done.append(fut.result())
                print(f"done {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"FAILED {name}: {e}; see results/{name}/sweep.log", flush=True)
    return done


if __name__ == "__main__":
    main()
