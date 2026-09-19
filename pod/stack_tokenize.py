"""Stream a Python code dataset, tokenise with the project's word-level regex tokeniser,
and write fixed-length windows to a uint16 memmap.

Outputs under --out (a prefix): <out>.bin (uint16, windows of n_ctx tokens, BOS first),
<out>_val.bin (same layout, from held-out files), <out>_meta.json (vocab, ids, n_ctx,
counts). Vocabulary: top --vocab-size tokens of the first --vocab-tokens tokens seen.

    HF_TOKEN=... uv run python pod/stack_tokenize.py --dataset bigcode/the-stack-dedup --config python \\
        --out /workspace/data/pool --target-tokens 2500000000
Fallback (not gated): --dataset codeparrot/github-code-clean --config Python-all --text-field code
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"\n|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")
SPECIAL = ["[PAD]", "[BOS]", "[UNK]"]


def stream(dataset: str, config: str | None, split: str, text_field: str):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True) if config else load_dataset(dataset, split=split, streaming=True)
    for row in ds:
        yield row[text_field]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bigcode/the-stack-dedup")
    p.add_argument("--config", default="python")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="content")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--vocab-tokens", type=int, default=200_000_000, help="tokens used to build the vocabulary")
    p.add_argument("--target-tokens", type=int, default=2_500_000_000)
    p.add_argument("--val-every", type=int, default=200, help="every k-th file goes to validation")
    p.add_argument("--val-windows", type=int, default=20000)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    body = a.n_ctx - 1

    # pass 1: vocabulary from the first vocab_tokens tokens (stream restarts for pass 2)
    counts: Counter = Counter()
    seen = 0
    for text in stream(a.dataset, a.config, a.split, a.text_field):
        toks = TOKEN_RE.findall(text)
        counts.update(toks)
        seen += len(toks)
        if seen >= a.vocab_tokens:
            break
    vocab = SPECIAL + [w for w, _ in counts.most_common(a.vocab_size - len(SPECIAL))]
    idx = {w: i for i, w in enumerate(vocab)}
    pad_id, bos_id, unk_id = 0, 1, 2
    print(f"vocabulary: {len(vocab)} tokens from {seen:,} tokens; UNK share on the sample {(1 - sum(counts[w] for w in vocab[3:]) / seen):.4f}", flush=True)

    # pass 2: windows
    n_train_windows = a.target_tokens // a.n_ctx
    train = np.memmap(str(a.out) + ".bin", dtype=np.uint16, mode="w+", shape=(n_train_windows, a.n_ctx))
    val = np.zeros((a.val_windows, a.n_ctx), dtype=np.uint16)
    nt = nv = files = unk = tot = 0
    for fi, text in enumerate(stream(a.dataset, a.config, a.split, a.text_field)):
        ids = [idx.get(t, unk_id) for t in TOKEN_RE.findall(text)]
        files += 1
        to_val = (fi % a.val_every == 0) and nv < a.val_windows
        for i in range(0, len(ids) - body + 1, body):
            w = [bos_id] + ids[i : i + body]
            unk += sum(1 for x in w if x == unk_id); tot += a.n_ctx
            if to_val:
                if nv < a.val_windows:
                    val[nv] = w; nv += 1
            elif nt < n_train_windows:
                train[nt] = w; nt += 1
        if nt >= n_train_windows and nv >= a.val_windows:
            break
        if files % 20000 == 0:
            print(f"files {files:,} train windows {nt:,}/{n_train_windows:,} val {nv:,}", flush=True)
    train.flush()
    if nt < n_train_windows:  # source exhausted: truncate the memmap to what was written
        del train
        arr = np.memmap(str(a.out) + ".bin", dtype=np.uint16, mode="r", shape=(n_train_windows, a.n_ctx))[:nt].copy()
        arr.tofile(str(a.out) + ".bin")
    val[:nv].tofile(str(a.out) + "_val.bin")
    meta = {"task": "text", "kind": "code", "vocab": vocab, "pad_id": pad_id, "bos_id": bos_id, "unk_id": unk_id, "n_ctx": a.n_ctx,
            "n_train_windows": int(nt), "n_val_windows": int(nv), "files": files, "frac_unk": unk / max(tot, 1),
            "config": {"dataset": a.dataset, "config": a.config, "vocab_size": len(vocab), "ctx_len": a.n_ctx}}
    Path(str(a.out) + "_meta.json").write_text(json.dumps(meta))
    print(f"wrote {nt:,} train windows ({nt * a.n_ctx / 1e9:.2f}B tokens), {nv:,} val windows; UNK {meta['frac_unk']:.4f}", flush=True)


if __name__ == "__main__":
    main()
