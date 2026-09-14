"""Real-text pool for Phase 3: TinyStories stories, word-level tokenizer built from the
corpus, one document per story truncated to `ctx_len` tokens. Same npz layout as the
synthetic tasks (train/val splits with tokens, target_mask, s_ids); target_mask is empty
because there is no task-defined target, and formation is measured by the prefix-matching
induction score (analysis/prefix_matching.py).

    uv run python -m data.textcorpus --src data/raw/TinyStoriesV2-GPT4-valid.txt --out datasets/text_pool.npz
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
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build(src: Path, ctx_len: int, vocab_size: int, n_val: int, min_tokens: int, seed: int, drop_ends: bool = False):
    rng = np.random.default_rng(seed)
    stories = [s.strip() for s in src.read_text(encoding="utf-8", errors="ignore").split("<|endoftext|>") if s.strip()]
    if drop_ends:  # a byte-range slice starts and ends mid-story
        stories = stories[1:-1]
    toks = [tokenize(s) for s in stories]
    toks = [t for t in toks if len(t) >= min_tokens]
    rng.shuffle(toks)
    counts = Counter(w for t in toks[n_val:] for w in t)
    vocab = SPECIAL + [w for w, _ in counts.most_common(vocab_size - len(SPECIAL))]
    idx = {w: i for i, w in enumerate(vocab)}

    def encode(words: list[str]) -> np.ndarray:
        ids = [BOS_ID] + [idx.get(w, UNK_ID) for w in words[: ctx_len - 1]]
        return np.array(ids + [PAD_ID] * (ctx_len - len(ids)), dtype=np.int64)

    def split(items):
        tokens = np.stack([encode(t) for t in items])
        return {"tokens": tokens, "target_mask": np.zeros(tokens.shape, bool), "s_ids": np.full(tokens.shape, -1, dtype=np.int64)}

    splits = {"train": split(toks[n_val:]), "val": split(toks[:n_val])}
    all_tok = splits["train"]["tokens"]
    body = all_tok[all_tok != PAD_ID]
    meta = {
        "task": "text",
        "config": {"src": str(src), "ctx_len": ctx_len, "vocab_size": len(vocab), "n_val": n_val, "min_tokens": min_tokens, "seed": seed},
        "vocab": vocab, "pad_id": PAD_ID, "bos_id": BOS_ID, "unk_id": UNK_ID,
        "stats": {
            "n_train": len(all_tok), "n_val": n_val, "frac_unk": float((body == UNK_ID).mean()),
            "frac_pad": float((all_tok == PAD_ID).mean()), "mean_story_tokens": float(np.mean([len(t) for t in toks])),
        },
    }
    return splits, meta


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ctx-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-ends", action="store_true", help="drop the first and last (partial) stories of a byte-range slice")
    a = p.parse_args(argv)
    splits, meta = build(a.src, a.ctx_len, a.vocab_size, a.n_val, a.min_tokens, a.seed, a.drop_ends)
    save_dataset(a.out, splits, meta)
    print(f"wrote {a.out}: train {splits['train']['tokens'].shape} val {splits['val']['tokens'].shape}")
    print("  " + json.dumps(meta["stats"]))
    print("  example:", " ".join(meta["vocab"][i] for i in splits["train"]["tokens"][0][:40]))
    return a.out


if __name__ == "__main__":
    main()
