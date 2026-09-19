"""Next-token loss on the positions where copying matters in code: second-and-later
occurrences within a document of low-frequency identifiers (vocabulary rank above
`--min-rank`, alphabetic tokens that are not Python keywords), compared with all other
positions. A copy head should help exactly there.

    uv run python -m analysis.identifier_loss --runs code_long_zipper_s0 code_long_random_s0 --step 24000
"""

from __future__ import annotations

import argparse
import json
import keyword
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.generator import load_dataset
from train.train import load_checkpoint


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--pool", type=Path, default=Path("datasets/code_pool.npz"))
    p.add_argument("--min-rank", type=int, default=500)
    p.add_argument("--n-docs", type=int, default=1000)
    a = p.parse_args(argv)
    splits, meta = load_dataset(a.pool)
    vocab = meta["vocab"]
    kw = set(keyword.kwlist) | {"self", "cls", "None", "True", "False"}
    is_ident = np.array([i >= a.min_rank and w[0].isalpha() and w not in kw and w != "[UNK]" for i, w in enumerate(vocab)])
    tok = torch.as_tensor(splits["val"]["tokens"][: a.n_docs])
    pad = meta["pad_id"]
    T = tok.shape[1]
    repeat_ident = np.zeros(tok.shape, bool)  # target is a low-freq identifier seen earlier in the doc
    first_ident = np.zeros(tok.shape, bool)   # target is a low-freq identifier seen for the first time
    for i in range(tok.shape[0]):
        seen = set()
        for t in range(1, T):
            x = int(tok[i, t])
            if x == pad:
                break
            if is_ident[x]:
                (repeat_ident if x in seen else first_ident)[i, t] = True
                seen.add(x)
    valid = (tok != pad).numpy()
    valid[:, 0] = False
    other = valid & ~repeat_ident & ~first_ident
    out = {}
    print(f"targets: {repeat_ident.sum()} repeated low-frequency identifiers, {first_ident.sum()} first occurrences, {other.sum()} other tokens")
    for run in a.runs:
        model, _ = load_checkpoint(Path("checkpoints", run, f"step_{a.step}.pt"))
        model.eval()
        with torch.no_grad():
            logits = model(tok)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tok[:, 1:].reshape(-1), reduction="none").view(tok.shape[0], T - 1).numpy()
        acc = (logits[:, :-1].argmax(-1) == tok[:, 1:]).numpy()
        r = {}
        for name, m in (("repeat_ident", repeat_ident), ("first_ident", first_ident), ("other", other), ("all", valid)):
            mm = m[:, 1:]
            r[name] = {"ce": float(ce[mm].mean()), "acc": float(acc[mm].mean())}
        out[run] = r
        print(f"{run} step {a.step}: " + " | ".join(f"{k}: CE {v['ce']:.3f} acc {v['acc']:.3f}" for k, v in r.items()))
    Path("results/identifier_loss.json").write_text(json.dumps({"step": a.step, "min_rank": a.min_rank, "runs": out}, indent=2))
    return out


if __name__ == "__main__":
    main()
