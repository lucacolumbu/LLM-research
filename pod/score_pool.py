"""Score a memmapped pool and build arm index files.

Gain: per-window zlib compression gain against a shuffled copy (multiprocessing).
Arms (each --frac of the pool): random_s{seed}; gain (top by gain); divgain (greedy by gain
in chunks of --chunk windows with --probe fixed probes per chunk drawn from the accepted set,
accept if NCD to every probe >= --ncd-min). Report: gain distribution, LZ77 match-length
histogram and mean pairwise NCD of pool vs each arm on samples.

    uv run python pod/score_pool.py --pool /workspace/data/pool --frac 0.2 --seeds 0 1 2
Writes <pool>_gain.npy, <pool>_arm_<name>.npy, <pool>_score_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.zipper import lz77_match_lengths, ncd, to_bytes

_W = None


def _init(path, n_ctx):
    global _W
    _W = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, n_ctx)


def _gain(i):
    d = np.asarray(_W[i][1:])  # drop BOS
    rng = np.random.default_rng(i)
    lx = len(zlib.compress(to_bytes(d), 9))
    ls = len(zlib.compress(to_bytes(rng.permutation(d)), 9))
    return 1.0 - lx / max(ls, 1)


def _ncd_ok(args):
    i, probes, ncd_min = args
    a = to_bytes(np.asarray(_W[i][1:]))
    return all(ncd(a, to_bytes(np.asarray(_W[j][1:]))) >= ncd_min for j in probes)


def match_stats(W, ids, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(ids, min(n, len(ids)), replace=False)
    lengths, longest, cov8, tot = [], [], 0, 0
    for i in sample:
        d = np.asarray(W[i][1:])
        m = lz77_match_lengths(d)
        lengths += m; longest.append(max(m) if m else 0); tot += len(d); cov8 += sum(x for x in m if x >= 8)
    arr = np.array(lengths) if lengths else np.zeros(0)
    bins = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 256]
    hist = np.histogram(arr, bins=bins)[0].tolist() if len(arr) else [0] * (len(bins) - 1)
    return {"n": len(sample), "bins": bins, "hist": hist, "median_match": float(np.median(arr)) if len(arr) else 0.0,
            "mean_longest": float(np.mean(longest)), "frac_tokens_in_matches_ge8": cov8 / max(tot, 1)}


def diversity(W, ids, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(ids, n_pairs); j = rng.choice(ids, n_pairs)
    return float(np.mean([ncd(to_bytes(np.asarray(W[a][1:])), to_bytes(np.asarray(W[b][1:]))) for a, b in zip(i, j) if a != b]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True, help="prefix: <pool>.bin and <pool>_meta.json")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ncd-min", type=float, default=0.75)
    p.add_argument("--probe", type=int, default=16)
    p.add_argument("--chunk", type=int, default=10000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    meta = json.loads(Path(str(a.pool) + "_meta.json").read_text())
    n_ctx = meta["n_ctx"]
    bin_path = str(a.pool) + ".bin"
    W = np.memmap(bin_path, dtype=np.uint16, mode="r").reshape(-1, n_ctx)
    N = len(W)
    n_arm = int(a.frac * N)
    gain_path = Path(str(a.pool) + "_gain.npy")
    if gain_path.exists():
        gain = np.load(gain_path)
    else:
        with Pool(a.workers, initializer=_init, initargs=(bin_path, n_ctx)) as pool:
            gain = np.array(pool.map(_gain, range(N), chunksize=2000), dtype=np.float32)
        np.save(gain_path, gain)
    print(f"pool {N:,} windows; gain mean {gain.mean():.3f}, 10/50/90th pct {np.percentile(gain, [10, 50, 90]).round(3).tolist()}", flush=True)
    order = np.argsort(-gain, kind="stable")
    arms = {"gain": np.sort(order[:n_arm])}
    rng = np.random.default_rng(0)
    for s in a.seeds:
        arms[f"random_s{s}"] = np.sort(np.random.default_rng(s).choice(N, n_arm, replace=False))
    div_path = Path(str(a.pool) + "_arm_divgain.npy")
    if div_path.exists():
        arms["divgain"] = np.load(div_path)
    else:
        accepted: list[int] = []
        rejected = 0
        pos = 0
        with Pool(a.workers, initializer=_init, initargs=(bin_path, n_ctx)) as pool:
            while len(accepted) < n_arm and pos < N:
                chunk = order[pos : pos + a.chunk]; pos += a.chunk
                if accepted:
                    probes = [accepted[k] for k in rng.choice(len(accepted), min(a.probe, len(accepted)), replace=False)]
                    ok = pool.map(_ncd_ok, [(int(i), probes, a.ncd_min) for i in chunk], chunksize=200)
                else:
                    ok = [True] * len(chunk)
                for i, o in zip(chunk, ok):
                    if o and len(accepted) < n_arm:
                        accepted.append(int(i))
                    elif not o:
                        rejected += 1
                print(f"divgain: scanned {pos:,} accepted {len(accepted):,} rejected {rejected:,}", flush=True)
        arms["divgain"] = np.sort(np.array(accepted))
    for name, ids in arms.items():
        np.save(str(a.pool) + f"_arm_{name}.npy", ids)
    report = {"n_windows": N, "n_arm": n_arm, "gain_pct": np.percentile(gain, [10, 50, 90]).tolist(),
              "pool": {"match": match_stats(W, np.arange(N)), "ncd": diversity(W, np.arange(N))}}
    for name, ids in arms.items():
        report[name] = {"mean_gain": float(gain[ids].mean()), "match": match_stats(W, ids), "ncd": diversity(W, ids)}
        print(f"{name:10s} gain {report[name]['mean_gain']:.3f}  tokens in matches>=8 {report[name]['match']['frac_tokens_in_matches_ge8']:.3f}  mean longest {report[name]['match']['mean_longest']:.1f}  NCD {report[name]['ncd']:.3f}", flush=True)
    Path(str(a.pool) + "_score_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
