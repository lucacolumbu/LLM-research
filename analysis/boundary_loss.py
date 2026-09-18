"""Loss around copy boundaries for lag-0 and lag-k induction models.

For every copied segment in validation documents: cross-entropy at each position within
the copy (j = 1 .. r-1, predicting the j-th copied token), at the first two post-copy
positions (predicting the tokens after the copy ends), and the "over-run" probability the
model assigns at the first post-copy position to the token that followed the *source*
segment (what a copy mechanism that does not know the copy has ended would predict).
Targets are split by whether the copy offset exceeds the model's lag.

    uv run python -m analysis.boundary_loss --run mlp_ind_lenlong_s0 --lag 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analysis.circuit import _step_of
from data.generator import load_dataset
from data.induction import need_positions
from train.train import load_checkpoint


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--n-docs", type=int, default=1000)
    a = p.parse_args(argv)
    tc = json.loads(Path("results", a.run, "train_config.json").read_text())
    splits, meta = load_dataset(Path(tc["dataset"]))
    tok = splits["val"]["tokens"][: a.n_docs]
    mask = splits["val"]["target_mask"][: a.n_docs]
    need = need_positions(tok, mask)
    model, _ = load_checkpoint(max(Path("checkpoints", a.run).glob("step_*.pt"), key=_step_of))
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(tok))
    lp = F.log_softmax(logits, -1).numpy()
    T = tok.shape[1]
    by_j: dict[str, dict[int, list[float]]] = {"long": {}, "short": {}}
    post: dict[str, dict[str, list[float]]] = {"long": {"ce+1": [], "ce+2": [], "overrun_p": [], "ce_first_copied": []}, "short": {"ce+1": [], "ce+2": [], "overrun_p": [], "ce_first_copied": []}}
    for i in range(len(tok)):
        pi = np.flatnonzero(mask[i])
        if len(pi) == 0:
            continue
        dst, r = int(pi[0]) - 1, len(pi) + 1
        src = int(need[i, pi[0]]) - 1
        if src < 0:
            continue
        group = "long" if (dst - src - 1) > a.lag else "short"
        by_j[group].setdefault(0, []).append(-lp[i, dst - 1, tok[i, dst]])  # first copied token, unpredictable
        for j in range(1, r):
            by_j[group].setdefault(j, []).append(-lp[i, dst + j - 1, tok[i, dst + j]])
        end = dst + r  # first post-copy position index
        if end < T:
            post[group]["ce+1"].append(-lp[i, end - 1, tok[i, end]])
            if src + r < T:
                post[group]["overrun_p"].append(float(np.exp(lp[i, end - 1, tok[i, src + r]])))
        if end + 1 < T:
            post[group]["ce+2"].append(-lp[i, end, tok[i, end + 1]])
    out = {"run": a.run, "lag": a.lag}
    uniform = float(np.log(meta["config"]["vocab_size"]))
    print(f"{a.run} (lag {a.lag}); uniform-over-vocab CE = {uniform:.2f} nats")
    for group in ("long", "short"):
        js = sorted(by_j[group])
        if not js:
            continue
        n = len(by_j[group][1]) if 1 in by_j[group] else 0
        row = {j: float(np.mean(by_j[group][j])) for j in js}
        out[group] = {"n_copies": n, "ce_by_j": row, **{k: float(np.mean(v)) if v else None for k, v in post[group].items()}}
        head = " ".join(f"j{j}:{row[j]:.2f}" for j in js[:8])
        tail = " ".join(f"j{j}:{row[j]:.2f}" for j in js[-3:]) if len(js) > 8 else ""
        print(f"  offset {'>' if group == 'long' else '<='} lag ({n} copies): CE within copy {head} ... {tail}")
        print(f"     post-copy: CE at +1 {out[group]['ce+1']:.2f}, at +2 {out[group]['ce+2']:.2f}; over-run prob (source continuation) at +1 {out[group]['overrun_p']:.3f} (chance {1/meta['config']['vocab_size']:.3f})")
    Path("results", a.run, "boundary_loss.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
