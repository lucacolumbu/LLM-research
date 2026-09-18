"""Accuracy on copied tokens as a function of the copy offset (dst - src) at a given
checkpoint. A one-layer "attend d back and copy" head predicts correctly exactly when the
offset equals d + 1, so it shows as a spike at one offset before any induction head exists.

    uv run python -m analysis.offset_accuracy --run mlp_ind_rep0.75_s0 --steps 2400 6000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.generator import load_dataset
from data.induction import need_positions
from train.train import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", nargs="+", type=int, required=True)
    p.add_argument("--n-docs", type=int, default=2000)
    a = p.parse_args(argv)
    tc = json.loads(Path("results", a.run, "train_config.json").read_text())
    splits, _ = load_dataset(Path(tc["dataset"]))
    tok = splits["val"]["tokens"][: a.n_docs]
    mask = splits["val"]["target_mask"][: a.n_docs]
    need = need_positions(tok, mask)
    b, pp = np.nonzero(mask)
    ok = need[b, pp] >= 0
    b, pp = b[ok], pp[ok]
    offset = pp - need[b, pp]  # dst - src for every target
    for step in a.steps:
        model, _ = load_checkpoint(Path("checkpoints", a.run, f"step_{step}.pt"))
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(tok))[:, :-1].argmax(-1).numpy()
        correct = pred[b, pp - 1] == tok[b, pp]
        accs = {int(o): float(correct[offset == o].mean()) for o in np.unique(offset) if (offset == o).sum() >= 30}
        overall = float(correct.mean())
        top = sorted(accs.items(), key=lambda kv: -kv[1])[:4]
        print(f"{a.run} step {step}: overall acc {overall:.3f}; top offsets by accuracy: " + ", ".join(f"o={o}: {v:.2f} (n={(offset == o).sum()})" for o, v in top))
        line = " ".join(f"{o}:{accs[o]:.2f}" for o in sorted(accs) if o <= 40)
        print(f"   acc by offset (2..40): {line}")


if __name__ == "__main__":
    main()
