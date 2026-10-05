"""Pure induction task: random token sequences with an embedded verbatim repeat.

Document: [BOS] followed by B = ctx_len - 1 uniformly random tokens, with one segment of
random length r in [2, r_max] copied verbatim from a random earlier position to a random
later position. Both the source-to-copy offset and the copy length vary per document. The target positions are the copied tokens
except the first one: predicting the next token after seeing a token for the second time
requires "find the previous occurrence of the current token and copy what followed it",
which is the induction head by construction. There is no fixed candidate set, so exclusion
cannot solve it. Two shortcuts had to be removed: with a fixed offset the task was solved
by a layer-0 positional head, and with a fixed copy length r the model built its induction
match on "the token r positions back" instead of the previous token.

Knobs (InductionConfig):
- repeat_frac   r_max = round(repeat_frac * B / 2): maximum copy length; r ~ U{2..r_max}
- vocab_size    active token types (<= MAX_VOCAB; the tokenizer is fixed at MAX_VOCAB)
- noise         P(a copied token is replaced by a random one): conditional entropy of
                the target given the context
- n_repeats     number of independent copied segments per document; each segment's
                maximum length is r_max // n_repeats, so the copied fraction stays
                roughly fixed while the *structure* (few long vs many short) changes
- repeat_frac_min  if > 0, each document draws its own fraction ~ U(repeat_frac_min,
                repeat_frac): a heterogeneous pool for selection experiments; the
                per-document fraction is stored as `doc_frac`
- length_dist   distribution of the copy length r in [2, r_cap]: "uniform" (default),
                "short_tail" (80% of documents draw r from [2, r_cap // 3], 20% from
                the full range: mostly short with a long tail, same maximum),
                "long" (r = r_cap always), or "geometric" (memoryless: constant hazard of
                the copy ending, parameterised for the uniform's mean but capped at r_cap,
                which lowers the realised mean by about 1.6 tokens). Separates "lag follows the maximum" from
                "lag follows the typical length".

Splits: train, val. `s_ids` holds the current token at each target (the token whose
previous occurrence must be found), so analysis.circuit's `attn_s` measures duplicate-
token attention and `attn_io` measures attention to the token to be copied.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.generator import save_dataset

MAX_VOCAB = 128
SPECIAL = ["[PAD]", "[BOS]"]
VOCAB_TOKENS = SPECIAL + [f"t{i}" for i in range(MAX_VOCAB)]
PAD_ID, BOS_ID = 0, 1


@dataclass
class InductionConfig:
    repeat_frac: float = 0.5
    vocab_size: int = 100
    noise: float = 0.0
    n_repeats: int = 1
    repeat_frac_min: float = 0.0
    length_dist: str = "uniform"
    ctx_len: int = 64
    n_train: int = 20_000
    n_val: int = 2_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.repeat_frac <= 1.0:
            raise ValueError("repeat_frac must be in (0, 1]")
        if not 2 <= self.vocab_size <= MAX_VOCAB:
            raise ValueError(f"vocab_size must be in [2, {MAX_VOCAB}]")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must be in [0, 1]")
        if self.repeat_len < 2:
            raise ValueError("repeat_frac too small for this ctx_len: fewer than 2 copied tokens")
        if 2 * self.repeat_len > self.ctx_len - 1:
            raise ValueError("repeat_frac too large: source and copy must both fit")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        if self.length_dist not in ("uniform", "short_tail", "long", "geometric"):
            raise ValueError("length_dist must be uniform, short_tail, long or geometric")
        if not 0.0 <= self.repeat_frac_min <= self.repeat_frac:
            raise ValueError("repeat_frac_min must be in [0, repeat_frac]")
        if self.repeat_frac_min > 0 and round(self.repeat_frac_min * (self.ctx_len - 1) / 2) < 2:
            raise ValueError("repeat_frac_min too small for this ctx_len")

    @property
    def repeat_len(self) -> int:
        return round(self.repeat_frac * (self.ctx_len - 1) / 2)


def generate_split(rng: np.random.Generator, cfg: InductionConfig, n: int) -> dict[str, np.ndarray]:
    B = cfg.ctx_len - 1
    lo, hi = len(SPECIAL), len(SPECIAL) + cfg.vocab_size
    body = rng.integers(lo, hi, size=(n, B))
    target_mask = np.zeros((n, B + 1), dtype=bool)
    doc_frac = np.full(n, cfg.repeat_frac)
    doc_copied = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if cfg.repeat_frac_min > 0:
            doc_frac[i] = rng.uniform(cfg.repeat_frac_min, cfg.repeat_frac)
        r_cap = max(2, round(doc_frac[i] * B / 2) // cfg.n_repeats)
        used = np.zeros(B, dtype=bool)
        for _ in range(cfg.n_repeats):
            if cfg.length_dist == "long":
                r = r_cap
            elif cfg.length_dist == "geometric":
                # memoryless lengths parameterised for the mean of uniform [2, r_cap]; the cap truncates
                # the tail, so the realised mean is about 1.6 below it
                mean_extra = (r_cap - 2) / 2
                r = min(2 + int(rng.geometric(1.0 / (mean_extra + 1.0))) - 1, r_cap)
            elif cfg.length_dist == "short_tail" and rng.random() < 0.8:
                r = int(rng.integers(2, max(2, r_cap // 3) + 1))
            else:
                r = int(rng.integers(2, r_cap + 1))
            for _attempt in range(200):
                src = int(rng.integers(0, B - 2 * r + 1))  # source segment start
                dst = int(rng.integers(src + r, B - r + 1))  # copy start, after the source ends
                if used[src : src + r].any() or used[dst : dst + r].any():
                    continue
                segment = body[i, src : src + r].copy()
                if cfg.noise > 0:
                    flip = rng.random(r) < cfg.noise
                    segment[flip] = rng.integers(lo, hi, size=flip.sum())
                body[i, dst : dst + r] = segment
                target_mask[i, 1 + dst + 1 : 1 + dst + r] = True  # copied tokens except the first
                used[src : src + r] = used[dst : dst + r] = True
                doc_copied[i] += r
                break
    tokens = np.concatenate([np.full((n, 1), BOS_ID), body], axis=1).astype(np.int64)
    s_ids = np.full_like(tokens, -1)
    b, p = target_mask.nonzero()
    s_ids[b, p] = tokens[b, p - 1]  # the current token at the prediction position
    return {"tokens": tokens, "target_mask": target_mask, "s_ids": s_ids, "doc_frac": doc_frac, "doc_copied": doc_copied}


def need_positions(tokens: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Position of the source token for every target (-1 where unknown). Reconstructed from
    the copy geometry; exact for single-repeat documents, first exact match otherwise."""
    need = np.full(tokens.shape, -1, dtype=np.int64)
    for i in range(tokens.shape[0]):
        pi = np.flatnonzero(target_mask[i])
        if len(pi) == 0:
            continue
        # split targets into runs (one per copied segment)
        runs = np.split(pi, np.flatnonzero(np.diff(pi) > 1) + 1)
        for run in runs:
            dst, r = int(run[0]) - 1, len(run) + 1
            seg = tokens[i, dst : dst + r]
            for s in range(1, dst - r + 1):
                if np.array_equal(tokens[i, s : s + r], seg):
                    need[i, run] = s + np.arange(1, r)
                    break
    return need


def generate(cfg: InductionConfig) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    splits = {"train": generate_split(rng, cfg, cfg.n_train), "val": generate_split(rng, cfg, cfg.n_val)}
    meta = {
        "task": "induction",
        "config": asdict(cfg),
        "vocab": VOCAB_TOKENS,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "repeat_len_max": cfg.repeat_len,
        "targets_per_doc_mean": float(splits["train"]["target_mask"].sum(1).mean()),
        "token_entropy_bits": math.log2(cfg.vocab_size),
        "stats": {
            s: {"frac_target": float(d["target_mask"].mean()), "copied_tokens_per_doc": float(d["doc_copied"].mean())}
            for s, d in splits.items()
        },
    }
    return splits, meta


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    for f in InductionConfig.__dataclass_fields__.values():
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = p.parse_args(argv)
    cfg = InductionConfig(**{k: v for k, v in vars(args).items() if k != "out"})
    splits, meta = generate(cfg)
    save_dataset(args.out, splits, meta)
    print(f"wrote {args.out}: r_max={cfg.repeat_len} targets/doc={meta['targets_per_doc_mean']:.1f} "
          f"train={splits['train']['tokens'].shape}")
    print("  " + json.dumps(meta["stats"]))
    return args.out


if __name__ == "__main__":
    main()
