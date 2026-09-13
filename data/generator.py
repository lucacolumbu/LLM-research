"""Synthetic two-name "who did what" dataset with independently controllable knobs.

Each sentence follows the Indirect Object Identification template:

    when A and B went to the PLACE , S gave a OBJ to IO .

where S is one of {A, B} and IO is the other. A document is a [BOS] followed by
as many sentences as fit in `ctx_len`, then [PAD].

Knobs (all independent, all in DataConfig):
- name_pool_size / name_zipf  -> name-pool entropy (memorise vs generalise)
- repetition_rate             -> P(a sentence reuses an earlier (A, B, S) triple
                                 in the same document), the known induction driver
- target_noise                -> P(IO is replaced by a random regular name): raises
                                 the conditional entropy of the target given context
- distractor_ratio            -> P(a filler phrase is inserted at each of two slots)

Generalisation splits. Every pool name is fully trained as input and as a predicted
token; what is held out is a *role* or a *combination*, never a token:
- heldout_io:    `n_heldout_io_names` pool names are, in training, always the subject
                 S when they occur in a pair, so they are never the answer. This split
                 makes them the IO. Only a content-independent copy mechanism passes.
- heldout_pairs: `heldout_pair_frac` of the regular-name pairs never co-occur in
                 training. This split is built from those pairs.
- heldout_io_leak (soft variant): with probability `heldout_io_leak`, a training
                 sentence whose pair contains a held-out name uses it as the IO after
                 all. Turns the held-out-IO metric into sample efficiency of the copy.

The tokenizer is a fixed word-level vocabulary shared by every condition.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# fmt: off
NAMES = [
    "Mary", "John", "Alice", "Bob", "Carol", "David", "Emma", "Frank",
    "Grace", "Henry", "Irene", "Jack", "Karen", "Liam", "Nora", "Oscar",
    "Paula", "Quinn", "Ruth", "Sam", "Tina", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Adam", "Beth", "Chris", "Dana", "Eric", "Fay",
    "Gary", "Hana", "Ivan", "Jill", "Kurt", "Lena", "Mark", "Nina",
    "Omar", "Pete", "Rosa", "Sean", "Tara", "Uma", "Vera", "Will",
    "Anna", "Ben", "Cara", "Dean", "Ella", "Finn", "Gina", "Hugo",
    "Ida", "Joel", "Kate", "Leo", "Mia", "Noah", "Olga", "Paul",
]
# fmt: on
PLACES = ["store", "park", "school", "garden", "beach", "market", "office", "library"]
OBJECTS = ["drink", "book", "ring", "letter", "ball", "kiss", "bone", "necklace"]
ADJECTIVES = ["sunny", "rainy", "cold", "quiet", "busy", "long", "late", "warm"]
TIMES = ["day", "morning", "afternoon", "evening"]
FUNCTION_WORDS = ["when", "and", "went", "to", "the", ",", "gave", "a", ".", "on"]
SPECIAL = ["[PAD]", "[BOS]"]

assert len(NAMES) == 64
MAX_SENTENCE_LEN = 9 + 4 + 4 + 7  # template + two fillers
SPLITS = ("train", "val", "heldout_pairs", "heldout_io")


class Vocab:
    """Fixed word-level vocabulary. Identical for every condition."""

    def __init__(self) -> None:
        self.tokens: list[str] = (
            SPECIAL + FUNCTION_WORDS + PLACES + OBJECTS + ADJECTIVES + TIMES + NAMES
        )
        assert len(set(self.tokens)) == len(self.tokens), "duplicate token"
        self.id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.id["[PAD]"]
        self.bos_id = self.id["[BOS]"]
        self.name_ids = np.array([self.id[n] for n in NAMES])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, words: list[str]) -> list[int]:
        return [self.id[w] for w in words]

    def decode(self, ids: list[int] | np.ndarray) -> str:
        return " ".join(self.tokens[int(i)] for i in ids)


VOCAB = Vocab()


@dataclass
class DataConfig:
    name_pool_size: int = 16  # distinct names in the pool, <= 64
    name_zipf: float = 0.0  # 0 = uniform pool; larger = more skewed frequencies
    repetition_rate: float = 0.5  # P(next sentence reuses an earlier triple in the doc)
    target_noise: float = 0.0  # P(IO replaced by a random regular name)
    distractor_ratio: float = 0.0  # P(filler phrase at each of two slots per sentence)
    n_heldout_io_names: int = 2  # pool names never used as IO in training
    heldout_pair_frac: float = 0.2  # fraction of regular pairs never co-occurring in training
    heldout_io_leak: float = 0.0  # soft variant: P(IO is the held-out name | pair contains one)
    ctx_len: int = 64
    n_train: int = 20_000
    n_val: int = 2_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.name_pool_size <= len(NAMES):
            raise ValueError(f"name_pool_size must be in [2, {len(NAMES)}]")
        if not 0 <= self.n_heldout_io_names <= self.name_pool_size - 2:
            raise ValueError("n_heldout_io_names must leave at least 2 regular names")
        if not 0.0 <= self.heldout_pair_frac < 1.0:
            raise ValueError("heldout_pair_frac must be in [0, 1)")
        if self.ctx_len < 1 + MAX_SENTENCE_LEN:
            raise ValueError(f"ctx_len must be >= {1 + MAX_SENTENCE_LEN}")
        for k in ("repetition_rate", "target_noise", "distractor_ratio", "heldout_io_leak"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f"{k} must be in [0, 1]")


def zipf_probs(n: int, exponent: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = ranks**-exponent
    return p / p.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class _Pool:
    """Name pool for one condition: ids, frequencies, held-out IO names and held-out pairs."""

    def __init__(self, cfg: DataConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        k = cfg.name_pool_size
        self.ids = VOCAB.name_ids[:k]
        self.probs = zipf_probs(k, cfg.name_zipf)
        h = cfg.n_heldout_io_names
        # held-out IO names take evenly spaced frequency ranks, not the rare tail
        held_idx = np.round(np.linspace(0, k - 1, h + 2)[1:-1]).astype(int) if h else np.array([], int)
        if len(set(held_idx.tolist())) != h:
            held_idx = np.arange(1, h + 1)
        self.heldout_io = np.array([int(self.ids[i]) for i in held_idx])
        reg_idx = [i for i in range(k) if i not in set(held_idx.tolist())]
        self.regular = np.array([int(self.ids[i]) for i in reg_idx])
        self.regular_probs = self.probs[reg_idx] / self.probs[reg_idx].sum()
        pairs = [frozenset((int(self.ids[i]), int(self.ids[j]))) for i, j in combinations(reg_idx, 2)]
        rng.shuffle(pairs)
        n_held = round(cfg.heldout_pair_frac * len(pairs))
        self.heldout_pairs = pairs[:n_held]
        self.heldout_pair_set = set(self.heldout_pairs)
        self.heldout_io_set = set(self.heldout_io.tolist())

    def _pick(self, rng: np.random.Generator, ids: np.ndarray, probs: np.ndarray | None, size=1, replace=True):
        return [int(x) for x in rng.choice(ids, size=size, replace=replace, p=probs)]

    def sample_triple(self, rng: np.random.Generator, split: str) -> tuple[int, int, int]:
        """Return (A, B, S) respecting the split's held-out rules."""
        for _ in range(10_000):
            if split in ("train", "val"):
                a, b = self._pick(rng, self.ids, self.probs, size=2, replace=False)
                a_h, b_h = a in self.heldout_io_set, b in self.heldout_io_set
                if (a_h and b_h) or frozenset((a, b)) in self.heldout_pair_set:
                    continue
                if a_h or b_h:
                    held, other = (a, b) if a_h else (b, a)
                    s = other if rng.random() < self.cfg.heldout_io_leak else held
                else:
                    s = a if rng.random() < 0.5 else b
                return a, b, s
            if split == "heldout_pairs":
                a, b = tuple(self.heldout_pairs[int(rng.integers(len(self.heldout_pairs)))])
                if rng.random() < 0.5:
                    a, b = b, a
                return a, b, (a if rng.random() < 0.5 else b)
            if split == "heldout_io":
                h = int(rng.choice(self.heldout_io))
                r = self._pick(rng, self.regular, self.regular_probs)[0]
                if frozenset((h, r)) in self.heldout_pair_set:
                    continue
                return (h, r, r) if rng.random() < 0.5 else (r, h, r)
            raise ValueError(f"unknown split {split!r}")
        raise RuntimeError(f"could not sample a valid pair for split {split!r}")

    def noise_io(self, rng: np.random.Generator) -> int:
        return self._pick(rng, self.regular, self.regular_probs)[0]

    def can_generate(self, split: str) -> bool:
        if split == "heldout_pairs":
            return len(self.heldout_pairs) > 0
        if split == "heldout_io":
            return len(self.heldout_io) > 0
        return True


def _filler(rng: np.random.Generator) -> list[str]:
    return ["on", "a", str(rng.choice(ADJECTIVES)), str(rng.choice(TIMES))]


def _sentence(
    rng: np.random.Generator,
    pool: _Pool,
    cfg: DataConfig,
    split: str,
    reuse: tuple[int, int, int] | None,
) -> tuple[list[int], int, int, tuple[int, int, int]]:
    """Return (token ids, index of IO within sentence, S id, (A, B, S) triple)."""
    a, b, s = reuse if reuse is not None else pool.sample_triple(rng, split)
    io = b if s == a else a
    if rng.random() < cfg.target_noise:
        io = pool.noise_io(rng)

    words: list[str | int] = []
    if rng.random() < cfg.distractor_ratio:
        words += _filler(rng)
    words += ["when", a, "and", b, "went", "to", "the", str(rng.choice(PLACES)), ","]
    if rng.random() < cfg.distractor_ratio:
        words += _filler(rng)
    words += [s, "gave", "a", str(rng.choice(OBJECTS)), "to"]
    io_index = len(words)
    words += [io, "."]
    ids = [w if isinstance(w, int) else VOCAB.id[w] for w in words]
    return ids, io_index, s, (a, b, s)


def _document(
    rng: np.random.Generator, pool: _Pool, cfg: DataConfig, split: str
) -> tuple[list[int], list[bool], list[int], int, int]:
    tokens = [VOCAB.bos_id]
    target_mask = [False]
    s_ids = [-1]
    history: list[tuple[int, int, int]] = []
    n_sentences = n_repeated = 0
    while True:
        reuse = None
        if history and rng.random() < cfg.repetition_rate:
            reuse = history[int(rng.integers(len(history)))]
        ids, io_index, s, triple = _sentence(rng, pool, cfg, split, reuse)
        if len(tokens) + len(ids) > cfg.ctx_len:
            break
        base = len(tokens)
        tokens += ids
        target_mask += [False] * len(ids)
        s_ids += [-1] * len(ids)
        target_mask[base + io_index] = True
        s_ids[base + io_index] = s
        history.append(triple)
        n_sentences += 1
        n_repeated += reuse is not None
    pad = cfg.ctx_len - len(tokens)
    tokens += [VOCAB.pad_id] * pad
    target_mask += [False] * pad
    s_ids += [-1] * pad
    return tokens, target_mask, s_ids, n_sentences, n_repeated


def generate_split(
    rng: np.random.Generator, pool: _Pool, cfg: DataConfig, split: str, n_docs: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not pool.can_generate(split):
        n_docs = 0
    tokens = np.zeros((n_docs, cfg.ctx_len), dtype=np.int64)
    target_mask = np.zeros((n_docs, cfg.ctx_len), dtype=bool)
    s_ids = np.full((n_docs, cfg.ctx_len), -1, dtype=np.int64)
    n_sent = n_rep = 0
    for i in range(n_docs):
        t, m, s, ns, nr = _document(rng, pool, cfg, split)
        tokens[i], target_mask[i], s_ids[i] = t, m, s
        n_sent += ns
        n_rep += nr
    stats = {
        "sentences_per_doc": n_sent / max(n_docs, 1),
        "frac_repeated_sentences": n_rep / max(n_sent, 1),
        "frac_pad": float((tokens == VOCAB.pad_id).mean()) if n_docs else 0.0,
        "frac_heldout_io_sentences": (
            float(np.isin(tokens[target_mask], pool.heldout_io).mean()) if n_docs else 0.0
        ),
    }
    return {"tokens": tokens, "target_mask": target_mask, "s_ids": s_ids}, stats


def generate(cfg: DataConfig) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Generate train / val / heldout_pairs / heldout_io splits and metadata."""
    rng = np.random.default_rng(cfg.seed)
    pool = _Pool(cfg, rng)
    splits: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        n = cfg.n_train if split == "train" else cfg.n_val
        splits[split], stats[split] = generate_split(rng, pool, cfg, split, n)

    meta = {
        "config": asdict(cfg),
        "vocab": VOCAB.tokens,
        "pad_id": VOCAB.pad_id,
        "bos_id": VOCAB.bos_id,
        "pool_name_ids": pool.ids.tolist(),
        "regular_name_ids": pool.regular.tolist(),
        "heldout_io_name_ids": pool.heldout_io.tolist(),
        "heldout_pairs": [sorted(p) for p in pool.heldout_pairs],
        "name_entropy_bits": entropy_bits(pool.probs),
        "max_name_entropy_bits": math.log2(cfg.name_pool_size),
        "stats": stats,
    }
    return splits, meta


def save_dataset(path: Path, splits: dict[str, dict[str, np.ndarray]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{split}__{k}": v for split, d in splits.items() for k, v in d.items()}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load_dataset(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        splits: dict[str, dict[str, np.ndarray]] = {}
        for key in f.files:
            if key == "meta":
                continue
            split, _, name = key.partition("__")
            splits.setdefault(split, {})[name] = f[key]
    return splits, meta


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output .npz path")
    for f in DataConfig.__dataclass_fields__.values():
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = p.parse_args(argv)

    cfg = DataConfig(**{k: v for k, v in vars(args).items() if k != "out"})
    splits, meta = generate(cfg)
    save_dataset(args.out, splits, meta)
    print(f"wrote {args.out}")
    print(f"  name entropy: {meta['name_entropy_bits']:.2f} / {meta['max_name_entropy_bits']:.2f} bits")
    print(f"  held-out IO names: {[VOCAB.tokens[i] for i in meta['heldout_io_name_ids']]}; "
          f"held-out pairs: {len(meta['heldout_pairs'])}")
    for split, st in meta["stats"].items():
        print(f"  {split:13s} {splits[split]['tokens'].shape} " + " ".join(f"{k}={v:.3f}" for k, v in st.items()))
    print("  example:", VOCAB.decode(splits["train"]["tokens"][0]).replace(" [PAD]", ""))
    return args.out


if __name__ == "__main__":
    main()
