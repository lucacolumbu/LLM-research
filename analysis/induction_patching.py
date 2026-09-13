"""The brief's formation definition on the induction task: activation patching.

Clean prompt: a validation document. Corrupt prompt: the same document with the source
segment replaced by random tokens, so the copied tokens can no longer be predicted from
context. Metric at each target position: log-probability of the correct token. Patching
copies the layer-1 attention outputs (all heads, `hook_z`) at the target query positions
from the clean run into the corrupt run; recovery = (patched - corrupt) / (clean - corrupt).
Formation step = first checkpoint with recovery >= 0.5. Also reports per-head recovery at
the final checkpoint.

    uv run python -m analysis.induction_patching --run ind_rep0.5_s0
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from analysis.circuit import _step_of
from data.generator import load_dataset
from data.induction import need_positions
from train.train import load_checkpoint


def make_corrupt(tokens: np.ndarray, target_mask: np.ndarray, vocab_size: int, seed: int = 0) -> np.ndarray:
    """Randomise every source token that a target copies from."""
    rng = np.random.default_rng(seed)
    need = need_positions(tokens, target_mask)
    corrupt = tokens.copy()
    b, p = target_mask.nonzero()
    src = need[b, p]
    ok = src >= 0
    corrupt[b[ok], src[ok]] = rng.integers(2, 2 + vocab_size, size=ok.sum())
    return corrupt


def _logprob(logits: torch.Tensor, b: torch.Tensor, q: torch.Tensor, tgt: torch.Tensor) -> float:
    return F.log_softmax(logits[b, q], -1)[torch.arange(len(b)), tgt].mean().item()


def _patch_hook(z: torch.Tensor, hook, clean_z: torch.Tensor, heads: list[int] | None, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    hs = slice(None) if heads is None else heads
    z[rows, cols, hs, :] = clean_z[rows, cols, hs, :]
    return z


@torch.no_grad()
def analyze_checkpoint(model, clean: np.ndarray, corrupt: np.ndarray, target_mask: np.ndarray, per_head: bool = False) -> dict[str, Any]:
    model.eval()
    dev = model.cfg.device
    tok_c = torch.as_tensor(clean).to(dev)
    tok_x = torch.as_tensor(corrupt).to(dev)
    mask = torch.as_tensor(target_mask).to(dev)
    b, p = mask.nonzero(as_tuple=True)
    q = p - 1
    tgt = tok_c[b, p]
    L = model.cfg.n_layers
    z_names = {f"blocks.{layer}.attn.hook_z" for layer in range(L)}
    clean_logits, cache = model.run_with_cache(tok_c, names_filter=lambda n: n in z_names)
    lp_clean = _logprob(clean_logits, b, q, tgt)
    lp_corrupt = _logprob(model(tok_x), b, q, tgt)
    denom = lp_clean - lp_corrupt
    last = L - 1
    name = f"blocks.{last}.attn.hook_z"

    def patched(heads: list[int] | None) -> float:
        hook = partial(_patch_hook, clean_z=cache[name], heads=heads, rows=b, cols=q)
        return _logprob(model.run_with_hooks(tok_x, fwd_hooks=[(name, hook)]), b, q, tgt)

    lp_patch = patched(None)
    out = {
        "n_targets": len(b),
        "acc_clean": (clean_logits[b, q].argmax(-1) == tgt).float().mean().item(),
        "logprob_clean": lp_clean, "logprob_corrupt": lp_corrupt, "logprob_patched": lp_patch,
        "recovery": (lp_patch - lp_corrupt) / denom if abs(denom) > 1e-6 else float("nan"),
    }
    if per_head:
        out["recovery_per_head"] = [
            (patched([h]) - lp_corrupt) / denom if abs(denom) > 1e-6 else float("nan") for h in range(model.cfg.n_heads)
        ]
    return out


def analyze_run(
    run: str, results_dir: Path = Path("results"), checkpoints_dir: Path = Path("checkpoints"), n_docs: int = 256, min_effect: float = 1.0
) -> dict[str, Any]:
    run_dir = results_dir / run
    tc = json.loads((run_dir / "train_config.json").read_text())
    splits, meta = load_dataset(Path(tc["dataset"]))
    tokens = splits["val"]["tokens"][:n_docs]
    mask = splits["val"]["target_mask"][:n_docs]
    corrupt = make_corrupt(tokens, mask, meta["config"]["vocab_size"])
    ckpts = sorted((checkpoints_dir / run).glob("step_*.pt"), key=_step_of)
    records = []
    for i, path in enumerate(ckpts):
        model, ckpt = load_checkpoint(path)
        rec = {"step": ckpt["step"], **analyze_checkpoint(model, tokens, corrupt, mask, per_head=(i == len(ckpts) - 1))}
        records.append(rec)
    with (run_dir / "induction_patching.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # formation requires a real clean-vs-corrupt effect (>= min_effect nats) and >= 50% recovery;
    # without the effect floor the ratio is noise before the model can copy at all
    summary = {
        "min_effect_nats": min_effect,
        "formation_step_patching": next(
            (r["step"] for r in records
             if r["step"] > 0 and (r["logprob_clean"] - r["logprob_corrupt"]) >= min_effect and r["recovery"] >= 0.5),
            None,
        ),
        "final": {k: records[-1][k] for k in ("acc_clean", "logprob_clean", "logprob_corrupt", "logprob_patched", "recovery", "recovery_per_head")},
    }
    (run_dir / "induction_patching_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n-docs", type=int, default=256)
    p.add_argument("--min-effect", type=float, default=1.0, help="clean minus corrupt log-prob floor (nats) for a formation call")
    a = p.parse_args(argv)
    s = analyze_run(a.run, n_docs=a.n_docs, min_effect=a.min_effect)
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
