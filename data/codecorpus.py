"""Real-code pool for Phase 3: Python source files chunked into fixed-length token windows.
Tokenizer: identifiers, numbers, string contents split into words, one token per
punctuation character, and an explicit newline token; vocabulary = the most frequent
tokens of the training split. Same npz layout as the other tasks; target_mask is empty.

    uv run python -m data.codecorpus --src .venv/lib/python3.14/site-packages --out datasets/code_pool.npz
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from data.generator import save_dataset

SPECIAL = ["[PAD]", "[BOS]", "[UNK]"]
PAD_ID, BOS_ID, UNK_ID = 0, 1, 2
TOKEN_RE = re.compile(r"\n|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build(src: Path, ctx_len: int, vocab_size: int, n_val: int, max_files: int, seed: int):
    rng = np.random.default_rng(seed)
    files = sorted(p for p in src.rglob("*.py") if p.stat().st_size > 2048)
    rng.shuffle(files)
    files = files[:max_files]
    docs: list[list[str]] = []
    for f in files:
        try:
            toks = tokenize(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        body = ctx_len - 1
        for i in range(0, len(toks) - body + 1, body):  # non-overlapping windows, drop the tail
            docs.append(toks[i : i + body])
    rng.shuffle(docs)
    counts = Counter(w for d in docs[n_val:] for w in d)
    vocab = SPECIAL + [w for w, _ in counts.most_common(vocab_size - len(SPECIAL))]
    idx = {w: i for i, w in enumerate(vocab)}

    def split(items):
        tokens = np.array([[BOS_ID] + [idx.get(w, UNK_ID) for w in d] for d in items], dtype=np.int64)
        return {"tokens": tokens, "target_mask": np.zeros(tokens.shape, bool), "s_ids": np.full(tokens.shape, -1, dtype=np.int64)}

    splits = {"train": split(docs[n_val:]), "val": split(docs[:n_val])}
    body = splits["train"]["tokens"][:, 1:]
    meta = {
        "task": "text",
        "config": {"src": str(src), "kind": "code", "ctx_len": ctx_len, "vocab_size": len(vocab), "n_val": n_val, "max_files": max_files, "seed": seed},
        "vocab": vocab, "pad_id": PAD_ID, "bos_id": BOS_ID, "unk_id": UNK_ID,
        "stats": {"n_files": len(files), "n_train": len(splits["train"]["tokens"]), "n_val": n_val, "frac_unk": float((body == UNK_ID).mean()), "frac_pad": 0.0},
    }
    return splits, meta


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ctx-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--n-val", type=int, default=5000)
    p.add_argument("--max-files", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    splits, meta = build(a.src, a.ctx_len, a.vocab_size, a.n_val, a.max_files, a.seed)
    save_dataset(a.out, splits, meta)
    print(f"wrote {a.out}: train {splits['train']['tokens'].shape} val {splits['val']['tokens'].shape}")
    print("  " + json.dumps(meta["stats"]))
    print("  example:", " ".join(meta["vocab"][i] for i in splits["train"]["tokens"][0][:40]).replace("\n", "\\n"))
    return a.out


if __name__ == "__main__":
    main()
