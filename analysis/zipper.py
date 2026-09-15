"""Tier 1: LZ77 "zipper" scores on token-ID sequences.

Conditional score  C(B|A) = (L(A+B) - L(A)) / L(B)   with L the LZ77-compressed length.
Low means B is largely explained by verbatim structure already in A.

Per-document compression gain  G(x) = 1 - L(x) / L(shuffle(x))   measures repeated
structure *within* a document. Shuffling keeps the unigram distribution, so Huffman
gains from skewed token frequencies do not count as structure.

Backends:
- "zlib"   DEFLATE = sliding-window LZ77 (32 KB window) + Huffman. Fast. Conditional
           scores reuse one compressor state for A, so scoring many candidates costs
           one compression of A plus one of each B.
- "python" Reference greedy sliding-window LZ77 with a fixed cost model and a
           configurable window. Slow; used for tests and sanity checks.

Pitfalls (from the brief): estimates converge slowly, only the last window of A can be
referenced, and scores are rankings, not absolutes. Keep B short relative to A.

    uv run python -m analysis.zipper --dataset datasets/base.npz [--update-results]
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generator import load_dataset
from train.train import results_lock


def strip_pad(tokens: np.ndarray, pad_id: int | None) -> np.ndarray:
    tokens = np.asarray(tokens).reshape(-1)
    return tokens if pad_id is None else tokens[tokens != pad_id]


def to_bytes(tokens: np.ndarray) -> bytes:
    tokens = np.asarray(tokens).reshape(-1)
    if len(tokens) == 0:
        return b""
    dtype = np.uint8 if tokens.max() < 256 else np.dtype("<u2")
    return tokens.astype(dtype).tobytes()


def _lz77_bits(seq: np.ndarray, window: int, vocab_bits: float, max_match: int = 255) -> float:
    """Greedy sliding-window LZ77 with a fixed cost model; returns encoded length in bits."""
    seq = [int(x) for x in seq]
    n = len(seq)
    lit_bits = 1.0 + vocab_bits
    match_bits = 1.0 + math.log2(max(window, 2)) + math.log2(max_match)
    index: dict[tuple[int, int], list[int]] = {}
    i = 0
    bits = 0.0
    while i < n:
        best = 0
        if i + 1 < n:
            for pos in reversed(index.get((seq[i], seq[i + 1]), [])):
                if i - pos > window:
                    break
                length = 2
                while i + length < n and length < max_match and seq[pos + length] == seq[i + length]:
                    length += 1
                best = max(best, length)
        if best >= 2 and match_bits < best * lit_bits:
            bits += match_bits
            step = best
        else:
            bits += lit_bits
            step = 1
        for j in range(i, i + step):
            if j + 1 < n:
                lst = index.setdefault((seq[j], seq[j + 1]), [])
                lst.append(j)
                if len(lst) > 64:
                    del lst[:-64]
        i += step
    return bits


def lz77_match_lengths(seq: np.ndarray, window: int = 4096, min_match: int = 2, max_match: int = 255) -> list[int]:
    """Greedy sliding-window LZ77 parse of a token sequence; returns the length of every match
    phrase (literals are not returned). The distribution of match lengths is the real-text
    analogue of the synthetic copy length."""
    seq = [int(x) for x in np.asarray(seq).reshape(-1)]
    n = len(seq)
    index: dict[tuple[int, int], list[int]] = {}
    i = 0
    out: list[int] = []
    while i < n:
        best = 0
        if i + 1 < n:
            for pos in reversed(index.get((seq[i], seq[i + 1]), [])):
                if i - pos > window:
                    break
                length = 2
                while i + length < n and length < max_match and seq[pos + length] == seq[i + length]:
                    length += 1
                best = max(best, length)
        step = best if best >= min_match else 1
        if best >= min_match:
            out.append(best)
        for j in range(i, i + step):
            if j + 1 < n:
                lst = index.setdefault((seq[j], seq[j + 1]), [])
                lst.append(j)
                if len(lst) > 64:
                    del lst[:-64]
        i += step
    return out


def match_length_stats(tokens: np.ndarray, pad_id: int, n_docs: int = 4000, seed: int = 0, bins=(2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 256)) -> dict:
    """Per-document LZ77 match-length distribution over a sample of documents: histogram of
    match lengths, mean and median match length, longest match per document, and the
    fraction of tokens covered by matches of at least 4 and at least 8 tokens."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tokens), min(n_docs, len(tokens)), replace=False)
    lengths: list[int] = []
    longest: list[int] = []
    cov4 = cov8 = total = 0
    for i in idx:
        doc = strip_pad(tokens[i], pad_id)
        m = lz77_match_lengths(doc)
        lengths += m
        longest.append(max(m) if m else 0)
        total += len(doc)
        cov4 += sum(x for x in m if x >= 4)
        cov8 += sum(x for x in m if x >= 8)
    arr = np.array(lengths) if lengths else np.zeros(0)
    hist, _ = np.histogram(arr, bins=list(bins)) if len(arr) else (np.zeros(len(bins) - 1, int), None)
    return {
        "n_docs": len(idx), "bins": list(bins), "hist": hist.tolist(),
        "matches_per_doc": float(len(arr) / len(idx)),
        "mean_match_len": float(arr.mean()) if len(arr) else 0.0,
        "median_match_len": float(np.median(arr)) if len(arr) else 0.0,
        "mean_longest_match": float(np.mean(longest)),
        "frac_tokens_in_matches_ge4": cov4 / max(total, 1),
        "frac_tokens_in_matches_ge8": cov8 / max(total, 1),
    }


def ncd(a: bytes, b: bytes, level: int = 9) -> float:
    """Normalized compression distance (L(AB) - min(L_A, L_B)) / max(L_A, L_B) with zlib."""
    la, lb = len(zlib.compress(a, level)), len(zlib.compress(b, level))
    lab = len(zlib.compress(a + b, level))
    return (lab - min(la, lb)) / max(max(la, lb), 1)


def corpus_diversity(tokens: np.ndarray, pad_id: int, n_pairs: int = 3000, seed: int = 0) -> dict:
    """Mean pairwise NCD over random document pairs: the second zipper quantity (how
    different documents are from each other), as opposed to per-document gain (how
    repetitive each document is internally). Nothing is subtracted: shared names and
    templates are meant to count."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    j = np.where(j == i, (j + 1) % n, j)
    d = np.array([ncd(to_bytes(strip_pad(tokens[a], pad_id)), to_bytes(strip_pad(tokens[b], pad_id))) for a, b in zip(i, j)])
    return {"mean_ncd": float(d.mean()), "std_ncd": float(d.std()), "n_pairs": n_pairs}


class Zipper:
    def __init__(
        self, backend: str = "zlib", level: int = 9, window: int = 4096, vocab_size: int = 256
    ) -> None:
        if backend not in ("zlib", "python"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.level = level
        self.window = window
        self.vocab_bits = math.log2(max(vocab_size, 2))

    def length_bits(self, tokens: np.ndarray) -> float:
        if self.backend == "zlib":
            return 8.0 * len(zlib.compress(to_bytes(tokens), self.level))
        return _lz77_bits(np.asarray(tokens).reshape(-1), self.window, self.vocab_bits)

    def conditional_many(self, candidates: list[np.ndarray], A: np.ndarray) -> np.ndarray:
        """C(B_i | A) for each candidate, compressing A only once."""
        A = np.asarray(A).reshape(-1)
        scores = np.zeros(len(candidates))
        if self.backend == "zlib":
            base = zlib.compressobj(self.level)
            head = base.compress(to_bytes(A))
            l_a = 8.0 * (len(head) + len(base.copy().flush()))
            for i, B in enumerate(candidates):
                c = base.copy()
                l_ab = 8.0 * (len(head) + len(c.compress(to_bytes(B)) + c.flush()))
                scores[i] = (l_ab - l_a) / max(self.length_bits(B), 1e-9)
        else:
            tail = A[-self.window :]  # only the window of A can be referenced
            l_a = self.length_bits(tail)
            for i, B in enumerate(candidates):
                l_ab = self.length_bits(np.concatenate([tail, np.asarray(B).reshape(-1)]))
                scores[i] = (l_ab - l_a) / max(self.length_bits(B), 1e-9)
        return scores

    def conditional(self, B: np.ndarray, A: np.ndarray) -> float:
        return float(self.conditional_many([B], A)[0])

    def compression_gain(
        self, tokens: np.ndarray, rng: np.random.Generator, n_shuffles: int = 1
    ) -> float:
        """1 - L(x) / L(shuffled x). Positive = repeated structure beyond unigram statistics."""
        tokens = np.asarray(tokens).reshape(-1)
        if len(tokens) < 2:
            return 0.0
        l_x = self.length_bits(tokens)
        l_shuf = np.mean([self.length_bits(rng.permutation(tokens)) for _ in range(n_shuffles)])
        return float(1.0 - l_x / max(l_shuf, 1e-9))


def dataset_zipper_score(
    splits: dict[str, dict[str, np.ndarray]],
    pad_id: int,
    backend: str = "zlib",
    n_docs: int | None = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Per-dataset tier-1 summary on the train split: per-doc gain and whole-corpus gain."""
    rng = np.random.default_rng(seed)
    zipper = Zipper(backend)
    docs = splits["train"]["tokens"][:n_docs]
    gains = np.array([zipper.compression_gain(strip_pad(d, pad_id), rng) for d in docs])
    corpus = strip_pad(docs, pad_id)
    return {
        "mean_doc_gain": float(gains.mean()),
        "std_doc_gain": float(gains.std()),
        "corpus_gain": zipper.compression_gain(corpus, rng),
        "n_docs": len(docs),
    }


def update_results(results_csv: Path, dataset: str, score: float) -> int:
    with results_lock(results_csv):
        df = pd.read_csv(results_csv)
        hit = df["dataset"] == dataset
        df.loc[hit, "zipper_score"] = score
        df.to_csv(results_csv, index=False)
    return int(hit.sum())


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", default="zlib", choices=["zlib", "python"])
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--update-results", action="store_true", help="fill zipper_score in results.csv")
    p.add_argument("--results-csv", type=Path, default=Path("results/results.csv"))
    args = p.parse_args(argv)

    splits, meta = load_dataset(Path(args.dataset))
    out = dataset_zipper_score(splits, meta["pad_id"], args.backend, args.n_docs)
    print(f"{args.dataset}: " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()
    ))
    if args.update_results:
        n = update_results(args.results_csv, args.dataset, out["mean_doc_gain"])
        print(f"updated zipper_score (mean_doc_gain) in {n} row(s) of {args.results_csv}")
    return out


if __name__ == "__main__":
    main()
