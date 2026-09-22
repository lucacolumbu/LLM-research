"""Final-checkpoint analysis on the pod for memmapped pools: in-context delta on rare
identifiers, name-swap control, and a chunked dump of every run's prefix-matching
trajectory and training log (lines <= 1500 chars so the log stream keeps them).

    uv run python pod/stack_analysis.py --pool /workspace/data/pool --runs stack_divgain_s0 stack_random_s0 ...
Writes /workspace/results/stack_analysis.json as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.circuit import _step_of
from analysis.incontext_delta import ce_matrix, deltas, summarize, tag_documents
from analysis.swap_control import gather_logprobs
from train.train import load_checkpoint

KW_RANK_SAMPLE = 1_000_000  # windows sampled from the pool to rank names by frequency


def emit(tag: str, obj) -> None:
    s = json.dumps(obj, separators=(",", ":"))
    for i in range(0, len(s), 1400):
        print(f"@@{tag} {i // 1400} {s[i:i + 1400]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dump-only", action="store_true", help="skip the analysis; print the saved results file and trajectories")
    a = p.parse_args()
    if a.dump_only:
        saved = a.results_dir / "stack_analysis.json"
        if saved.exists():
            emit("RESULTS", json.loads(saved.read_text()))
        dump_runs(a)
        print("STACK DUMP DONE", flush=True)
        return
    meta = json.loads(Path(a.pool + "_meta.json").read_text())
    vocab, pad, n_ctx = meta["vocab"], meta["pad_id"], meta["n_ctx"]
    pool = np.memmap(a.pool + ".bin", dtype=np.uint16, mode="r").reshape(-1, n_ctx)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(pool), min(KW_RANK_SAMPLE, len(pool)), replace=False))
    counts = np.bincount(np.asarray(pool[sample]).reshape(-1), minlength=len(vocab))
    from analysis.incontext_delta import BUILTINS, KW

    is_name = [bool(w) and (w[0].isalpha() or w[0] == "_") and w not in KW and w not in BUILTINS and not w.startswith("[") for w in vocab]
    ids = [i for i in range(len(vocab)) if is_name[i]]
    order = sorted(ids, key=lambda i: -counts[i])
    ranks = {tok: r for r, tok in enumerate(order)}
    val = np.fromfile(a.pool + "_val.bin", dtype=np.uint16).reshape(-1, n_ctx)[: a.n_docs].astype(np.int64)
    docs = tag_documents(val, vocab, pad, ranks, a.top_k)
    n_pairs = sum(len(d["idents"]) for d in docs)
    print(f"{len(val)} val windows; rare identifier pairs {n_pairs}; common pairs {sum(len(d['common']) for d in docs)}", flush=True)
    rare_pool = np.array([t for t, r in ranks.items() if a.top_k <= r < a.top_k + 3000])
    pairs, swapped = [], []
    for i, d in enumerate(docs):
        present = set(val[i].tolist())
        for x, q1, q2 in d["idents"]:
            y = int(rng.choice(rare_pool))
            while y in present:
                y = int(rng.choice(rare_pool))
            doc = val[i].copy(); doc[q1] = y
            pairs.append((i, x, y, q1, q2)); swapped.append(doc)
    swapped = np.stack(swapped) if swapped else np.zeros((0, n_ctx), dtype=np.int64)
    tv = torch.as_tensor(val)
    out = {"n_docs": len(val), "n_pairs": n_pairs, "top_k": a.top_k, "runs": {}}
    for run in a.runs:
        ck = max((a.checkpoints_dir / run).glob("step_*.pt"), key=_step_of)
        model, _ = load_checkpoint(ck, device=a.device)
        model.eval()
        with torch.no_grad():
            ce = ce_matrix(model, tv.to(a.device))
            d = deltas(ce, docs)
            res = {"checkpoint": ck.name, "delta": summarize(d), "common": summarize(deltas(ce, docs, "common"))}
            for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 256)):
                sub = d[(d.dist > lo) & (d.dist <= hi)]
                if len(sub):
                    res[f"dist_{lo}_{hi}"] = summarize(sub)
            if len(pairs):
                orig = val[[i for i, x, y, q1, q2 in pairs]]
                pos = np.array([q2 - 1 for i, x, y, q1, q2 in pairs])
                xs = np.array([x for i, x, y, q1, q2 in pairs]); ys = np.array([y for i, x, y, q1, q2 in pairs])
                lx, ly = gather_logprobs(model, orig, pos, xs, ys)
                lx_s, ly_s = gather_logprobs(model, swapped, pos, xs, ys)
                res["swap"] = {"x_drop": float((lx - lx_s).mean()), "y_gain": float((ly_s - ly).mean()), "frac_prefers_y_after_swap": float((ly_s > lx_s).mean())}
        out["runs"][run] = res
        print(f"{run} [{ck.name}]: CE1 {res['delta']['ce1']:.3f} CE2 {res['delta']['ce2']:.3f} delta {res['delta']['delta_mean']:+.3f} (frac>0 {res['delta']['frac_positive']:.2f}); common delta {res['common']['delta_mean']:+.3f}; swap x_drop {res.get('swap', {}).get('x_drop', float('nan')):+.2f} y_gain {res.get('swap', {}).get('y_gain', float('nan')):+.2f}", flush=True)
        emit(f"ANALYSIS {run}", res)
    (a.results_dir / "stack_analysis.json").write_text(json.dumps(out, indent=1))
    dump_runs(a)
    print("STACK ANALYSIS DONE", flush=True)


def dump_runs(a) -> None:
    """Full trajectories and training logs, chunked; tolerant of truncated log lines."""
    for run in a.runs:
        pm = a.results_dir / run / "prefix_matching_summary.json"
        if pm.exists():
            emit(f"PM {run}", json.loads(pm.read_text()).get("trajectory"))
        lg = a.results_dir / run / "log.jsonl"
        if lg.exists():
            rows = []
            for line in lg.read_text().splitlines():
                try:
                    r = json.loads(line)
                    rows.append([r["step"], round(r["train_loss"], 4), round(r["val_loss"], 4)])
                except (ValueError, KeyError):
                    continue
            emit(f"LOG {run}", rows)


if __name__ == "__main__":
    main()
