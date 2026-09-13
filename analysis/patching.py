"""Activation patching, head ablation, faithfulness and sharpness on single-sentence
IOI prompts, across checkpoints.

Prompts:  [BOS] when A and B went to the PLACE , S gave a OBJ to IO .
Corrupt:  the subject slot holds the other name, so the correct answer flips.
Prediction is read at the position of "to" (index 14); logit diff is IO minus S of the
clean prompt.

Per checkpoint (validation-style prompts, regular names, training pairs only):
- per-head patching: copy one head's output at the answer position from the clean run
  into the corrupted run; recovery = (ld_patched - ld_corrupt) / (ld_clean - ld_corrupt)
- circuit heads: the top-k heads by recovery at the final checkpoint (k = 2 by default),
  held fixed across checkpoints
- formation step (brief definition): first checkpoint where patching the circuit heads
  recovers >= 50% of the clean-vs-corrupt logit difference
- ablation of the circuit heads (mean over the prompt batch, and zero): the exclusion
  story predicts accuracy collapses to ~0.5
- keep-only-k curve: keep the top-k heads by recovery, mean-ablate every other head.
  faithfulness = accuracy with only the circuit heads kept, relative to the full model;
  sharpness = smallest k that recovers 90% of the full logit difference
- context bonus: IO logit minus mean logit of pool names absent from the prompt, and
  which components produce it (from analysis.circuit with pool_ids)

    uv run python -m analysis.patching --run pool16_s0 [--update-results]
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from analysis.circuit import _step_of, analyze_checkpoint
from data.generator import OBJECTS, PLACES, VOCAB, load_dataset
from train.train import load_checkpoint, results_lock

Q = 14  # index of "to": the answer is predicted here
P = 15  # index of the IO token


def build_prompts(meta: dict[str, Any], n: int, seed: int = 0, split: str = "val") -> dict[str, np.ndarray]:
    """Single-sentence prompts. `val`: regular names, training pairs. `heldout_io`: a
    held-out name as IO with a regular subject."""
    rng = np.random.default_rng(seed)
    regular = list(meta["regular_name_ids"])
    held = list(meta["heldout_io_name_ids"])
    heldout_pairs = {frozenset(p) for p in meta["heldout_pairs"]}
    clean, corrupt, ios, ss = [], [], [], []
    while len(clean) < n:
        if split == "val":
            a, b = (int(x) for x in rng.choice(regular, size=2, replace=False))
            s = a if rng.random() < 0.5 else b
        elif split == "heldout_io":
            if not held:
                break
            h, r = int(rng.choice(held)), int(rng.choice(regular))
            a, b = (h, r) if rng.random() < 0.5 else (r, h)
            s = r
        else:
            raise ValueError(split)
        if frozenset((a, b)) in heldout_pairs:
            continue
        io = b if s == a else a
        place, obj = str(rng.choice(PLACES)), str(rng.choice(OBJECTS))
        base = ["[BOS]", "when", a, "and", b, "went", "to", "the", place, ",", s, "gave", "a", obj, "to", io, "."]
        ids = [w if isinstance(w, int) else VOCAB.id[w] for w in base]
        clean.append(ids)
        bad = list(ids)
        bad[10] = io  # swap the subject: now S is the answer
        corrupt.append(bad)
        ios.append(io)
        ss.append(s)
    tokens = np.array(clean, dtype=np.int64).reshape(-1, len(base))
    mask = np.zeros_like(tokens, dtype=bool)
    s_ids = np.full_like(tokens, -1)
    if len(tokens):
        mask[:, P] = True
        s_ids[:, P] = ss
    return {
        "tokens": tokens,
        "corrupt": np.array(corrupt, dtype=np.int64).reshape(-1, len(base)),
        "io": np.array(ios, dtype=np.int64),
        "s": np.array(ss, dtype=np.int64),
        "target_mask": mask,
        "s_ids": s_ids,
    }


def _metrics(logits: torch.Tensor, io: torch.Tensor, s: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, Q]
    ar = torch.arange(len(io))
    return (pred[ar, io] - pred[ar, s]).mean().item(), (pred.argmax(-1) == io).float().mean().item()


def _z_hook_ablate(z: torch.Tensor, hook, heads: list[int], value: torch.Tensor | None) -> torch.Tensor:
    """value: per-position means [pos, head, d_head] (mean ablation) or None (zero ablation)."""
    for h in heads:
        z[:, :, h, :] = 0.0 if value is None else value[:, h, :]
    return z


def _z_hook_patch(z: torch.Tensor, hook, heads: list[int], clean_z: torch.Tensor) -> torch.Tensor:
    for h in heads:
        z[:, Q, h, :] = clean_z[:, Q, h, :]
    return z


@torch.no_grad()
def run_ablated(
    model: HookedTransformer, tokens: torch.Tensor, heads: list[tuple[int, int]], means: dict[int, torch.Tensor] | None
) -> torch.Tensor:
    """Logits with the given heads ablated at every position (per-position mean over the prompt
    batch if `means` given, else zero). Prompts share one template, so per-position means are
    an in-distribution reference; a single mean over all positions is not."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    hooks = [
        (f"blocks.{layer}.attn.hook_z", partial(_z_hook_ablate, heads=hs, value=None if means is None else means[layer]))
        for layer, hs in by_layer.items()
    ]
    return model.run_with_hooks(tokens, fwd_hooks=hooks)


@torch.no_grad()
def run_patched(
    model: HookedTransformer, corrupt: torch.Tensor, heads: list[tuple[int, int]], clean_cache
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    hooks = [
        (f"blocks.{layer}.attn.hook_z", partial(_z_hook_patch, heads=hs, clean_z=clean_cache[f"blocks.{layer}.attn.hook_z"]))
        for layer, hs in by_layer.items()
    ]
    return model.run_with_hooks(corrupt, fwd_hooks=hooks)


@torch.no_grad()
def analyze_prompts(
    model: HookedTransformer,
    prompts: dict[str, np.ndarray],
    circuit_heads: list[tuple[int, int]] | None = None,
    ranking: list[tuple[int, int]] | None = None,
    pool_ids: list[int] | None = None,
    k_circuit: int = 2,
    recovery_target: float = 0.8,
) -> dict[str, Any]:
    model.eval()
    dev = model.cfg.device
    L, H = model.cfg.n_layers, model.cfg.n_heads
    clean = torch.as_tensor(prompts["tokens"]).to(dev)
    corrupt = torch.as_tensor(prompts["corrupt"]).to(dev)
    io, s = torch.as_tensor(prompts["io"]).to(dev), torch.as_tensor(prompts["s"]).to(dev)
    z_names = {f"blocks.{layer}.attn.hook_z" for layer in range(L)}
    clean_logits, cache = model.run_with_cache(clean, names_filter=lambda n: n in z_names)
    ld_clean, acc_clean = _metrics(clean_logits, io, s)
    ld_corrupt, acc_corrupt = _metrics(model(corrupt), io, s)
    means = {layer: cache[f"blocks.{layer}.attn.hook_z"].mean(0) for layer in range(L)}  # [pos, H, d_head]
    denom = ld_clean - ld_corrupt if abs(ld_clean - ld_corrupt) > 1e-6 else float("nan")

    recovery = np.zeros((L, H))
    for layer in range(L):
        for head in range(H):
            ld_p, _ = _metrics(run_patched(model, corrupt, [(layer, head)], cache), io, s)
            recovery[layer, head] = (ld_p - ld_corrupt) / denom
    if ranking is None:
        flat = np.argsort(-recovery, axis=None)
        ranking = [(int(i // H), int(i % H)) for i in flat]
    if circuit_heads is None:
        circuit_heads = ranking[:k_circuit]
        if k_circuit <= 0:  # adaptive: smallest k whose joint patch recovers >= recovery_target
            for k in range(1, min(L * H, 8) + 1):
                ld_k, _ = _metrics(run_patched(model, corrupt, ranking[:k], cache), io, s)
                if (ld_k - ld_corrupt) / denom >= recovery_target:
                    break
            circuit_heads = ranking[:k]

    ld_p, acc_p = _metrics(run_patched(model, corrupt, circuit_heads, cache), io, s)
    out: dict[str, Any] = {
        "n_prompts": len(clean),
        "ld_clean": ld_clean, "acc_clean": acc_clean, "ld_corrupt": ld_corrupt, "acc_corrupt": acc_corrupt,
        "recovery_per_head": recovery.tolist(),
        "circuit_heads": circuit_heads,
        "circuit_recovery": (ld_p - ld_corrupt) / denom,
        "circuit_patched_acc": acc_p,
    }
    for mode, m in (("mean", means), ("zero", None)):
        ld_a, acc_a = _metrics(run_ablated(model, clean, circuit_heads, m), io, s)
        out[f"ablate_circuit_{mode}_ld"], out[f"ablate_circuit_{mode}_acc"] = ld_a, acc_a

    keep_curve = []
    all_heads = [(layer, head) for layer in range(L) for head in range(H)]
    for k in range(L * H + 1):
        keep = set(ranking[:k])
        drop = [hd for hd in all_heads if hd not in keep]
        ld_k, acc_k = _metrics(run_ablated(model, clean, drop, means), io, s) if drop else (ld_clean, acc_clean)
        keep_curve.append({"k": k, "ld": ld_k, "acc": acc_k})
    out["keep_only_curve"] = keep_curve
    k_circuit = len(circuit_heads)
    out["faithfulness"] = keep_curve[k_circuit]["acc"] / acc_clean if acc_clean > 0 else float("nan")
    out["faithfulness_ld"] = keep_curve[k_circuit]["ld"] / ld_clean if ld_clean > 0 else float("nan")
    out["sharpness"] = next((c["k"] for c in keep_curve if c["ld"] >= 0.9 * ld_clean), None)

    split = {"tokens": prompts["tokens"], "target_mask": prompts["target_mask"], "s_ids": prompts["s_ids"]}
    dla = analyze_checkpoint(model, split, max_docs=len(prompts["tokens"]), pool_ids=pool_ids)
    for key in ("ctx_bonus", "components_ctx", "components_ld", "components_io", "attn_io", "attn_s", "dla_ctx"):
        if key in dla:
            out[key] = dla[key]
    return out


def analyze_run(
    run: str,
    results_dir: Path = Path("results"),
    checkpoints_dir: Path = Path("checkpoints"),
    n_prompts: int = 256,
    k_circuit: int = 2,
    update_results: bool = False,
) -> dict[str, Any]:
    run_dir = results_dir / run
    tc = json.loads((run_dir / "train_config.json").read_text())
    _, meta = load_dataset(Path(tc["dataset"]))
    pool = meta["pool_name_ids"]
    prompts = {sp: build_prompts(meta, n_prompts, seed=0, split=sp) for sp in ("val", "heldout_io")}
    ckpts = sorted((checkpoints_dir / run).glob("step_*.pt"), key=_step_of)

    # fix the circuit and ranking from the final checkpoint, then walk every checkpoint
    model, _ = load_checkpoint(ckpts[-1])
    final = analyze_prompts(model, prompts["val"], pool_ids=pool, k_circuit=k_circuit)
    rec = np.array(final["recovery_per_head"])
    flat = np.argsort(-rec, axis=None)
    ranking = [(int(i // rec.shape[1]), int(i % rec.shape[1])) for i in flat]
    circuit = [tuple(h) for h in final["circuit_heads"]]  # k_circuit heads, or adaptive when k_circuit <= 0

    records = []
    for path in ckpts:
        model, ckpt = load_checkpoint(path)
        r: dict[str, Any] = {"step": ckpt["step"]}
        r["val"] = analyze_prompts(model, prompts["val"], circuit, ranking, pool)
        if len(prompts["heldout_io"]["tokens"]):
            r["heldout_io"] = analyze_prompts(model, prompts["heldout_io"], circuit, ranking, pool)
        records.append(r)
    with (run_dir / "patching.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    steps = [r["step"] for r in records]
    formation = next((st for st, r in zip(steps, records) if st > 0 and r["val"]["circuit_recovery"] >= 0.5), None)
    fin = records[-1]["val"]
    summary = {
        "circuit_heads": [f"L{layer}H{head}" for layer, head in circuit],
        "formation_step_patching": formation,
        "faithfulness": fin["faithfulness"],
        "faithfulness_ld": fin["faithfulness_ld"],
        "sharpness": fin["sharpness"],
        "final_val": {k: fin[k] for k in (
            "acc_clean", "ld_clean", "ld_corrupt", "circuit_recovery", "ablate_circuit_mean_acc",
            "ablate_circuit_mean_ld", "ablate_circuit_zero_acc", "ctx_bonus")},
        "final_val_components_ctx": fin.get("components_ctx"),
        "final_val_recovery_per_head": fin["recovery_per_head"],
        "final_val_keep_only_curve": fin["keep_only_curve"],
    }
    if "heldout_io" in records[-1]:
        h = records[-1]["heldout_io"]
        summary["final_heldout_io"] = {k: h[k] for k in (
            "acc_clean", "ld_clean", "circuit_recovery", "ablate_circuit_mean_acc", "ablate_circuit_mean_ld", "ctx_bonus")}
        summary["final_heldout_io_components_ctx"] = h.get("components_ctx")
    (run_dir / "patching_summary.json").write_text(json.dumps(summary, indent=2))

    if update_results:
        csv = results_dir / "results.csv"
        with results_lock(csv):
            df = pd.read_csv(csv)
            hit = df["run"] == run
            df.loc[hit, "formation_step"] = formation
            df.loc[hit, "faithfulness"] = fin["faithfulness"]
            df.loc[hit, "sharpness"] = fin["sharpness"]
            df.to_csv(csv, index=False)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--k-circuit", type=int, default=2, help="heads in the circuit; 0 = smallest set recovering 80%")
    p.add_argument("--update-results", action="store_true", help="fill formation_step, faithfulness, sharpness")
    a = p.parse_args(argv)
    summary = analyze_run(a.run, a.results_dir, a.checkpoints_dir, a.n_prompts, a.k_circuit, a.update_results)
    brief = {k: v for k, v in summary.items() if not k.startswith("final_val_")}
    print(json.dumps(brief, indent=2))
    return summary


if __name__ == "__main__":
    main()
