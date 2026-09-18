"""Layer-0 relative-offset attention profiles across training. For each checkpoint and
each layer-0 head: mean attention from query position q to q-d for d = 0..max_d, over
positions inside copies and outside them separately, plus the layer-1 lag profile. Tells
whether a "token k+1 back" head exists before the lag-k induction head appears.

    uv run python -m analysis.l0_offsets --run mlp_ind_rep0.75_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.circuit import _step_of
from data.generator import load_dataset
from data.induction import need_positions
from train.train import load_checkpoint


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--n-docs", type=int, default=256)
    p.add_argument("--max-d", type=int, default=40)
    p.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    a = p.parse_args(argv)
    tc = json.loads(Path("results", a.run, "train_config.json").read_text())
    splits, _ = load_dataset(Path(tc["dataset"]))
    tok = torch.as_tensor(splits["val"]["tokens"][: a.n_docs])
    mask = splits["val"]["target_mask"][: a.n_docs]
    need = need_positions(splits["val"]["tokens"][: a.n_docs], mask)
    T = tok.shape[1]
    in_copy = torch.zeros(tok.shape, dtype=torch.bool)
    for i in range(len(tok)):
        pi = np.flatnonzero(mask[i])
        if len(pi):
            in_copy[i, pi[0] - 1 : pi[-1] + 1] = True  # query positions whose prediction is a copied token
    ckpts = sorted(Path("checkpoints", a.run).glob("step_*.pt"), key=_step_of)[:: a.every]
    records = []
    for path in ckpts:
        model, ckpt = load_checkpoint(path)
        model.eval()
        H = model.cfg.n_heads
        with torch.no_grad():
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n.endswith("attn.hook_pattern"))
        pat0 = cache["blocks.0.attn.hook_pattern"]  # [n, H, q, k]
        q_idx = torch.arange(T)
        prof = {"copy": np.zeros((H, a.max_d + 1)), "other": np.zeros((H, a.max_d + 1))}
        for d in range(a.max_d + 1):
            qs = q_idx[d:]
            att = pat0[:, :, qs, qs - d]  # [n, H, |qs|]
            for grp, m in (("copy", in_copy[:, d:]), ("other", ~in_copy[:, d:])):
                w = m[:, None, :].float()  # [n, 1, |qs|]
                prof[grp][:, d] = ((att * w).sum((0, 2)) / w.sum((0, 2)).clamp(min=1)).numpy()
        rec = {"step": ckpt["step"]}
        for grp in ("copy", "other"):
            best_d = prof[grp].argmax(1)
            rec[f"l0_{grp}_best_offset"] = best_d.tolist()
            rec[f"l0_{grp}_best_mass"] = prof[grp].max(1).tolist()
            rec[f"l0_{grp}_profile"] = prof[grp].tolist()
        # layer-1 relative-offset profile at copy-target queries (positional seed in layer 1?)
        pat1 = cache["blocks.1.attn.hook_pattern"]
        prof1 = np.zeros((H, a.max_d + 1))
        for d in range(a.max_d + 1):
            qs = q_idx[d:]
            att = pat1[:, :, qs, qs - d]
            w = in_copy[:, d:][:, None, :].float()
            prof1[:, d] = ((att * w).sum((0, 2)) / w.sum((0, 2)).clamp(min=1)).numpy()
        rec["l1_copy_best_offset"] = prof1.argmax(1).tolist()
        rec["l1_copy_best_mass"] = prof1.max(1).tolist()
        b, pp = np.nonzero(mask)
        nd = need[b, pp]
        ok = nd >= 0
        b, pp, nd = b[ok], pp[ok], nd[ok]
        lagm = np.zeros((H, a.max_d + 1))
        for o in range(a.max_d + 1):
            k = nd + o
            good = k <= pp - 1
            if good.any():
                lagm[:, o] = pat1[b[good], :, pp[good] - 1, k[good]].mean(0).numpy()
        rec["l1_best_lag"] = lagm.argmax(1).tolist()
        rec["l1_best_lag_mass"] = lagm.max(1).tolist()
        records.append(rec)
        c0 = " ".join(f"H{h}:{rec['l0_copy_best_offset'][h]}({rec['l0_copy_best_mass'][h]:.2f})" for h in range(H))
        l1 = " ".join(f"H{h}:{rec['l1_best_lag'][h]}({rec['l1_best_lag_mass'][h]:.2f})" for h in range(H))
        c1 = " ".join(f"H{h}:{rec['l1_copy_best_offset'][h]}({rec['l1_copy_best_mass'][h]:.2f})" for h in range(H))
        print(f"step {rec['step']:5d} | L0 offset back inside copies: {c0} | L1 offset back: {c1} | L1 lag: {l1}")
    with Path("results", a.run, "l0_offsets.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return records


if __name__ == "__main__":
    main()
