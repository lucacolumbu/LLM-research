"""In-context delta on rare identifiers: CE at an identifier's first occurrence in a
document minus CE at its second occurrence. Isolates the copy head's contribution from
distribution quality (CE1 orders like validation loss; the delta should not).

Tagging (cached once per validation document, shared across arms): NAME = token matching
[A-Za-z_]\\w*, not a Python keyword, not a builtin, not inside a string literal (quote
parity within the window). Rare = name not among the top-K names by frequency in the full
pool (K = 2000 by default; 500 and 5000 as sensitivity). Per document, identifiers with
>= 2 occurrences contribute (CE1, CE2) for their first and second occurrence only.

Controls: distance bins between the two occurrences (<=32, 32-64, 64-128); shuffled-context
(tokens before the second occurrence permuted, first occurrence still present; one
identifier per document); non-identifier baseline (same delta for repeated keywords and
punctuation).

    uv run python -m analysis.incontext_delta --arms zipper divgain mix random --seeds 0 1 2 --step 8000
    uv run python -m analysis.incontext_delta ... --trajectory   (every checkpoint, mean delta only)
"""

from __future__ import annotations

import argparse
import builtins
import json
import keyword
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.circuit import _step_of
from data.generator import load_dataset
from train.train import load_checkpoint

KW = set(keyword.kwlist) | set(keyword.softkwlist) | {"self", "cls"}
BUILTINS = set(dir(builtins))


def tag_documents(tokens: np.ndarray, vocab: list[str], pad_id: int, name_rank: dict[int, int], top_k: int):
    """Per document: list of (ident_id, q1, q2) for rare identifiers with >= 2 occurrences, and
    list of (tok, q1, q2) for repeated common tokens (keywords / punctuation)."""
    is_name = np.array([bool(w) and (w[0].isalpha() or w[0] == "_") and w.replace("_", "a").isalnum() and w not in KW and w not in BUILTINS and not w.startswith("[") for w in vocab])
    is_common = np.array([(w in KW) or (len(w) == 1 and not w.isalnum() and w not in "\"'") for w in vocab])
    quote_ids = {i for i, w in enumerate(vocab) if w in ('"', "'")}
    docs = []
    for i in range(tokens.shape[0]):
        in_str = False
        first_name: dict[int, int] = {}
        second_name: dict[int, int] = {}
        first_common: dict[int, int] = {}
        second_common: dict[int, int] = {}
        for q in range(1, tokens.shape[1]):
            x = int(tokens[i, q])
            if x == pad_id:
                break
            if x in quote_ids:
                in_str = not in_str
                continue
            if in_str:
                continue
            if is_name[x] and name_rank.get(x, 10**9) >= top_k:
                if x not in first_name:
                    first_name[x] = q
                elif x not in second_name:
                    second_name[x] = q
            elif is_common[x]:
                if x not in first_common:
                    first_common[x] = q
                elif x not in second_common:
                    second_common[x] = q
        docs.append({
            "idents": [(x, first_name[x], second_name[x]) for x in second_name],
            "common": [(x, first_common[x], second_common[x]) for x in second_common],
        })
    return docs


def name_ranks(pool_tokens: np.ndarray, vocab: list[str]) -> dict[int, int]:
    counts = np.bincount(pool_tokens.reshape(-1), minlength=len(vocab))
    is_name = [bool(w) and (w[0].isalpha() or w[0] == "_") and w not in KW and w not in BUILTINS and not w.startswith("[") for w in vocab]
    ids = [i for i in range(len(vocab)) if is_name[i]]
    order = sorted(ids, key=lambda i: -counts[i])
    return {tok: r for r, tok in enumerate(order)}


@torch.no_grad()
def ce_matrix(model, tokens: torch.Tensor, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(tokens), batch):
        t = tokens[i : i + batch]
        logits = model(t)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1), reduction="none").view(len(t), -1)
        out.append(ce.cpu().numpy())
        del logits, ce
    return np.concatenate(out)  # [n, T-1]; ce[:, q-1] is the loss of predicting token q


def deltas(ce: np.ndarray, docs, key: str = "idents"):
    rows = []
    for i, d in enumerate(docs):
        for x, q1, q2 in d[key]:
            rows.append((i, x, q1, q2, ce[i, q1 - 1], ce[i, q2 - 1]))
    df = pd.DataFrame(rows, columns=["doc", "tok", "q1", "q2", "ce1", "ce2"])
    df["delta"] = df.ce1 - df.ce2
    df["dist"] = df.q2 - df.q1
    return df


def summarize(df: pd.DataFrame) -> dict:
    return {"n": len(df), "ce1": float(df.ce1.mean()), "ce2": float(df.ce2.mean()), "delta_mean": float(df.delta.mean()),
            "delta_median": float(df.delta.median()), "frac_positive": float((df.delta > 0).mean())}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["zipper", "divgain", "mix", "random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--prefix", default="code_big")
    p.add_argument("--pool", type=Path, default=Path("datasets/code_pool.npz"))
    p.add_argument("--step", type=int, default=8000)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--sensitivity", nargs="*", type=int, default=[500, 5000])
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--trajectory", action="store_true", help="mean delta at every checkpoint (writes results/incontext_delta_trajectory.csv)")
    p.add_argument("--n-shuffle-docs", type=int, default=600)
    a = p.parse_args(argv)

    splits, meta = load_dataset(a.pool)
    vocab, pad = meta["vocab"], meta["pad_id"]
    ranks = name_ranks(splits["train"]["tokens"], vocab)
    val = splits["val"]["tokens"][: a.n_docs]
    tv = torch.as_tensor(val)
    tags = {k: tag_documents(val, vocab, pad, ranks, k) for k in [a.top_k, *a.sensitivity]}
    docs = tags[a.top_k]
    n_id = sum(len(d["idents"]) for d in docs)
    print(f"{a.n_docs} validation documents; rare identifiers (not top-{a.top_k}) with >= 2 occurrences: {n_id}; repeated common tokens: {sum(len(d['common']) for d in docs)}")
    for k in a.sensitivity:
        print(f"   sensitivity top-{k}: {sum(len(d['idents']) for d in tags[k])} identifier pairs")

    runs = [(arm, s, f"{a.prefix}_{arm}_s{s}") for arm in a.arms for s in a.seeds]
    if a.trajectory:
        rows = []
        for arm, s, run in runs:
            for ck in sorted(Path("checkpoints", run).glob("step_*.pt"), key=_step_of):
                model, _ = load_checkpoint(ck)
                model.eval()
                ce = ce_matrix(model, tv)
                d = deltas(ce, docs)
                rows.append({"arm": arm, "seed": s, "step": _step_of(ck), **summarize(d)})
            print(f"trajectory done: {run}")
        pd.DataFrame(rows).to_csv("results/incontext_delta_trajectory.csv", index=False)
        return

    table, dist_rows, shuf_rows, common_rows, sens_rows = [], [], [], [], []
    rng = np.random.default_rng(0)
    for arm, s, run in runs:
        model, _ = load_checkpoint(Path("checkpoints", run, f"step_{a.step}.pt"))
        model.eval()
        ce = ce_matrix(model, tv)
        d = deltas(ce, docs)
        table.append({"arm": arm, "seed": s, **summarize(d)})
        for lo, hi in ((0, 32), (32, 64), (64, 128)):
            sub = d[(d.dist > lo) & (d.dist <= hi)]
            if len(sub):
                dist_rows.append({"arm": arm, "seed": s, "bin": f"{lo}-{hi}", **summarize(sub)})
        common_rows.append({"arm": arm, "seed": s, **summarize(deltas(ce, docs, "common"))})
        for k in a.sensitivity:
            sens_rows.append({"arm": arm, "seed": s, "top_k": k, **summarize(deltas(ce, tags[k]))})
        # shuffled-context control: one rare identifier per document, permute tokens 1..q2-1
        cand = [(i, d_["idents"][0]) for i, d_ in enumerate(docs) if d_["idents"]][: a.n_shuffle_docs]
        shuf = val[[i for i, _ in cand]].copy()
        for j, (i, (x, q1, q2)) in enumerate(cand):
            perm = rng.permutation(np.arange(1, q2))
            shuf[j, 1:q2] = val[i, perm]
        ce_s = ce_matrix(model, torch.as_tensor(shuf))
        ce1 = np.array([ce[i, q1 - 1] for i, (x, q1, q2) in cand])
        ce2 = np.array([ce[i, q2 - 1] for i, (x, q1, q2) in cand])
        ce2s = np.array([ce_s[j, q2 - 1] for j, (i, (x, q1, q2)) in enumerate(cand)])
        shuf_rows.append({"arm": arm, "seed": s, "n": len(cand), "ce1": float(ce1.mean()), "ce2": float(ce2.mean()), "ce2_shuffled": float(ce2s.mean()),
                          "delta": float((ce1 - ce2).mean()), "delta_shuffled": float((ce1 - ce2s).mean()), "frac_delta_retained": float((ce1 - ce2s).mean() / max((ce1 - ce2).mean(), 1e-9))})
        print(f"{run}: CE1 {table[-1]['ce1']:.3f} CE2 {table[-1]['ce2']:.3f} delta {table[-1]['delta_mean']:+.3f} (median {table[-1]['delta_median']:+.3f}, frac>0 {table[-1]['frac_positive']:.2f}) | shuffled-context delta {shuf_rows[-1]['delta_shuffled']:+.3f}")

    out = {"step": a.step, "top_k": a.top_k, "n_docs": a.n_docs, "table": table, "distance": dist_rows, "shuffled": shuf_rows, "common": common_rows, "sensitivity": sens_rows}
    Path("results/incontext_delta.json").write_text(json.dumps(out, indent=2))
    T = pd.DataFrame(table)
    print("\nper arm (mean over seeds):")
    print(T.groupby("arm")[["ce1", "ce2", "delta_mean", "delta_median", "frac_positive"]].mean().round(3).loc[a.arms].to_string())
    print("\ndistance control (mean delta by bin):")
    print(pd.DataFrame(dist_rows).groupby(["arm", "bin"])["delta_mean"].mean().unstack().round(3).loc[a.arms].to_string())
    print("\nshuffled-context control (mean over seeds):")
    print(pd.DataFrame(shuf_rows).groupby("arm")[["delta", "delta_shuffled", "frac_delta_retained"]].mean().round(3).loc[a.arms].to_string())
    print("\nnon-identifier baseline (repeated keywords/punctuation):")
    print(pd.DataFrame(common_rows).groupby("arm")[["ce1", "ce2", "delta_mean"]].mean().round(3).loc[a.arms].to_string())
    print("\nsensitivity to the rarity threshold (mean delta):")
    print(pd.DataFrame(sens_rows).groupby(["arm", "top_k"])["delta_mean"].mean().unstack().round(3).loc[a.arms].to_string())


if __name__ == "__main__":
    main()
