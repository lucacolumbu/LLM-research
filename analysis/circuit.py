"""Head-level circuit measurements at indirect-object positions, across checkpoints.

For every IO target (the answer is predicted at the position of "to"), and for
every attention head, this measures:

- attn_io / attn_s   attention from the query position to every earlier occurrence
                     of the IO name / of the subject name. A name-mover head attends
                     to the IO; an S-inhibition head attends to the subject and pushes
                     its logit down. Formation shows up here before it wins at the output.
- dla_ld / dla_io    direct logit attribution of the head's output to the IO-minus-S
                     logit difference / to the IO logit itself.

Plus the same attribution for the non-head components: token and positional
embeddings, per-layer attention and MLP biases, and the pure "prior" term
(final LayerNorm bias and unembed bias), which is where a "this token is never
an answer" preference lives. The decomposition is exact: components + bias equal
the model's logit difference, and `decomposed_ld` is reported next to `logit_diff`
so that can be checked.

    uv run python -m analysis.circuit --run base_s0
writes results/<run>/circuit.jsonl (one record per checkpoint), circuit_summary.json
and trajectory.png.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from data.generator import load_dataset
from train.train import load_checkpoint


def _proj(x: torch.Tensor, d: torch.Tensor, scale: torch.Tensor, w_ln: torch.Tensor) -> torch.Tensor:
    """Push a residual component [n, ..., d_model] through the final LayerNorm onto direction d [n, d_model]."""
    x = x - x.mean(-1, keepdim=True)
    x = x / scale.view(scale.shape[0], *([1] * (x.ndim - 2)), 1)
    return torch.einsum("n...d,nd->n...", x * w_ln, d)


@torch.no_grad()
def analyze_checkpoint(
    model: HookedTransformer,
    split: dict[str, np.ndarray],
    max_docs: int = 512,
    batch_size: int = 128,
    pool_ids: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    """Per-head and per-component measurements at IO positions.

    With `pool_ids`, also decomposes the *context bonus*: the IO logit minus the mean
    logit of pool names absent from the context. That is the task-specific signal an
    exclusion-only circuit could not produce."""
    tokens_np = split["tokens"][:max_docs]
    mask_np = split["target_mask"][:max_docs]
    s_np = split["s_ids"][:max_docs]
    if len(tokens_np) == 0:
        return {}
    model.eval()
    L, H = model.cfg.n_layers, model.cfg.n_heads
    dev = model.cfg.device
    w_ln, b_ln = model.ln_final.w, model.ln_final.b
    W_U, b_U, W_O, b_O = model.W_U, model.b_U, model.W_O, model.b_O
    has_mlp = not getattr(model.cfg, "attn_only", False)
    has_pos = getattr(model.cfg, "positional_embedding_type", "standard") == "standard"
    hooks = {"hook_embed", "ln_final.hook_scale"} | ({"hook_pos_embed"} if has_pos else set())
    for layer in range(L):
        hooks |= {f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_z"}
        if has_mlp:
            hooks.add(f"blocks.{layer}.hook_mlp_out")

    heads = {k: torch.zeros(L, H) for k in ("attn_io", "attn_s", "dla_ld", "dla_io", "dla_ctx")}
    comp_ld: dict[str, float] = defaultdict(float)
    comp_io: dict[str, float] = defaultdict(float)
    comp_ctx: dict[str, float] = defaultdict(float)
    n_total = correct = 0
    ld_sum = io_logit_sum = ctx_bonus_sum = 0.0
    pool = torch.as_tensor(np.asarray(pool_ids), device=dev) if pool_ids is not None else None

    for i in range(0, len(tokens_np), batch_size):
        tok = torch.as_tensor(tokens_np[i : i + batch_size]).to(dev)
        mask = torch.as_tensor(mask_np[i : i + batch_size]).to(dev)
        s_ids = torch.as_tensor(s_np[i : i + batch_size]).to(dev)
        logits, cache = model.run_with_cache(tok, names_filter=lambda name: name in hooks)
        b, p = mask.nonzero(as_tuple=True)
        q = p - 1
        io, s = tok[b, p], s_ids[b, p]
        n = len(b)
        ar = torch.arange(n, device=dev)
        ctx = tok[b]
        before = torch.arange(ctx.shape[1], device=dev)[None, :] < p[:, None]
        io_mask = ((ctx == io[:, None]) & before).float()
        s_mask = ((ctx == s[:, None]) & before).float()

        pred = logits[b, q]
        correct += (pred.argmax(-1) == io).sum().item()
        ld_sum += (pred[ar, io] - pred[ar, s]).sum().item()
        io_logit_sum += pred[ar, io].sum().item()

        dir_ld = (W_U[:, io] - W_U[:, s]).T  # [n, d_model]
        dir_io = W_U[:, io].T
        scale = cache["ln_final.hook_scale"][b, q]  # [n, 1]
        if pool is not None:
            present = (ctx[:, :, None] == pool[None, None, :]).any(1)  # [n, P]
            absent = (~present).float()
            weights = absent / absent.sum(1, keepdim=True).clamp(min=1)
            dir_ctx = dir_io - weights @ W_U[:, pool].T  # IO minus mean absent-name unembed
            ctx_bonus_sum += (pred[ar, io] - (weights * pred[:, pool]).sum(1)).sum().item()

        comps = {"embed": cache["hook_embed"][b, q]}
        if has_pos:
            comps["pos_embed"] = cache["hook_pos_embed"][b, q]
        for layer in range(L):
            z = cache[f"blocks.{layer}.attn.hook_z"][b, q]  # [n, H, d_head]
            head_out = torch.einsum("nhe,hed->nhd", z, W_O[layer])
            heads["dla_ld"][layer] += _proj(head_out, dir_ld, scale, w_ln).sum(0).cpu()
            heads["dla_io"][layer] += _proj(head_out, dir_io, scale, w_ln).sum(0).cpu()
            if pool is not None:
                heads["dla_ctx"][layer] += _proj(head_out, dir_ctx, scale, w_ln).sum(0).cpu()
            comps[f"attn_bias_{layer}"] = b_O[layer].expand(n, -1)
            if has_mlp:
                comps[f"mlp_{layer}"] = cache[f"blocks.{layer}.hook_mlp_out"][b, q]
            pat = cache[f"blocks.{layer}.attn.hook_pattern"][b, :, q, :]  # [n, H, ctx]
            heads["attn_io"][layer] += (pat * io_mask[:, None, :]).sum(-1).sum(0).cpu()
            heads["attn_s"][layer] += (pat * s_mask[:, None, :]).sum(-1).sum(0).cpu()
        for name, x in comps.items():
            comp_ld[name] += _proj(x, dir_ld, scale, w_ln).sum().item()
            comp_io[name] += _proj(x, dir_io, scale, w_ln).sum().item()
            if pool is not None:
                comp_ctx[name] += _proj(x, dir_ctx, scale, w_ln).sum().item()
        comp_ld["bias"] += (b_ln @ dir_ld.T + b_U[io] - b_U[s]).sum().item()
        comp_io["bias"] += (b_ln @ dir_io.T + b_U[io]).sum().item()
        if pool is not None:
            comp_ctx["bias"] += (b_ln @ dir_ctx.T + b_U[io] - weights @ b_U[pool]).sum().item()
        n_total += n

    model.train()
    out: dict[str, Any] = {
        "n_targets": n_total,
        "io_acc": correct / n_total,
        "logit_diff": ld_sum / n_total,
        "io_logit": io_logit_sum / n_total,
    }
    for k, v in heads.items():
        out[k] = (v / n_total).tolist()
    out["components_ld"] = {k: v / n_total for k, v in comp_ld.items()}
    out["components_io"] = {k: v / n_total for k, v in comp_io.items()}
    out["components_ld"]["heads"] = float(heads["dla_ld"].sum() / n_total)
    out["components_io"]["heads"] = float(heads["dla_io"].sum() / n_total)
    out["decomposed_ld"] = float(sum(out["components_ld"].values()))
    if pool is not None:
        out["ctx_bonus"] = ctx_bonus_sum / n_total
        out["components_ctx"] = {k: v / n_total for k, v in comp_ctx.items()}
        out["components_ctx"]["heads"] = float(heads["dla_ctx"].sum() / n_total)
        out["decomposed_ctx"] = float(sum(out["components_ctx"].values()))
    else:
        del out["dla_ctx"]
    return out


def _step_of(path: Path) -> int:
    return int(path.stem.split("_")[1])


def summarize(records: list[dict[str, Any]], attn_threshold: float = 0.5) -> dict[str, Any]:
    """Name-mover candidates, attention-based formation step and logit-diff crossing step."""
    steps = [r["step"] for r in records]
    final_val = records[-1]["val"]
    dla = np.array(final_val["dla_ld"])
    order = np.dstack(np.unravel_index(np.argsort(-dla, axis=None), dla.shape))[0]
    top = [(int(layer), int(head)) for layer, head in order[:2]]
    gen_split = "heldout_io" if records[-1].get("heldout_io") else "val"

    def cand_max(r: dict[str, Any], key: str) -> float:
        return max(r[key][layer][head] for layer, head in top)

    def first_step(pred) -> int | None:
        return next((s for s, r in zip(steps, records) if r.get(gen_split) and pred(r[gen_split])), None)

    # upward zero crossing of the logit difference, ignoring the random-init value at step 0
    lds = [r[gen_split]["logit_diff"] if r.get(gen_split) else float("nan") for r in records]
    crossing = next(
        (steps[i] for i in range(1, len(lds)) if lds[i] > 0 and lds[i - 1] <= 0), None
    )

    # sustained crossing: first checkpoint from which at least 75% of the remaining
    # checkpoints have a positive logit difference (robust to hovering around zero)
    sustained = next(
        (steps[i] for i in range(1, len(lds))
         if np.mean([x > 0 for x in lds[i:]]) >= 0.75 and lds[i] > 0),
        None,
    )

    return {
        "steps": steps,
        "name_mover_candidates": top,
        "sustained_crossing_step_logit_diff": sustained,
        "generalisation_split": gen_split,
        # formation: first checkpoint at which one of the final top-attribution heads puts
        # >= threshold attention on the IO / on S (any-head variants for reference)
        "formation_step_attn_io": first_step(lambda r: cand_max(r, "attn_io") >= attn_threshold),
        "formation_step_attn_s": first_step(lambda r: cand_max(r, "attn_s") >= attn_threshold),
        "formation_step_attn_s_anyhead": first_step(lambda r: max(map(max, r["attn_s"])) >= attn_threshold),
        "final_candidate_attention": {
            f"L{layer}H{head}": {
                "attn_io": records[-1][gen_split]["attn_io"][layer][head],
                "attn_s": records[-1][gen_split]["attn_s"][layer][head],
            }
            for layer, head in top
        },
        "attn_threshold": attn_threshold,
        "crossing_step_logit_diff": crossing,
        "min_logit_diff": (min(lds[1:]), steps[1 + lds[1:].index(min(lds[1:]))]) if len(lds) > 1 else None,
        "final": {
            split: {k: records[-1][split][k] for k in ("io_acc", "logit_diff", "io_logit")}
            for split in ("val", "heldout_pairs", "heldout_io")
            if records[-1].get(split)
        },
        "final_components_ld": {s: records[-1][s]["components_ld"] for s in ("val", gen_split) if records[-1].get(s)},
    }


def analyze_run(
    run: str,
    results_dir: Path = Path("results"),
    checkpoints_dir: Path = Path("checkpoints"),
    max_docs: int = 512,
    attn_threshold: float = 0.5,
    plot: bool = True,
) -> dict[str, Any]:
    run_dir = results_dir / run
    tc = json.loads((run_dir / "train_config.json").read_text())
    splits, _ = load_dataset(Path(tc["dataset"]))
    ckpts = sorted((checkpoints_dir / run).glob("step_*.pt"), key=_step_of)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {checkpoints_dir / run}")

    records = []
    for path in ckpts:
        model, ckpt = load_checkpoint(path)
        rec: dict[str, Any] = {"step": ckpt["step"]}
        for split in ("val", "heldout_pairs", "heldout_io"):
            rec[split] = analyze_checkpoint(model, splits[split], max_docs) if split in splits else {}
        records.append(rec)
    with (run_dir / "circuit.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary = summarize(records, attn_threshold)
    (run_dir / "circuit_summary.json").write_text(json.dumps(summary, indent=2))
    if plot:
        from analysis.plots import plot_trajectory

        plot_trajectory(records, summary, run, run_dir / "trajectory.png")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--max-docs", type=int, default=512)
    p.add_argument("--attn-threshold", type=float, default=0.5)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--summary-only", action="store_true", help="rebuild summary and plot from circuit.jsonl")
    a = p.parse_args(argv)
    if a.summary_only:
        run_dir = a.results_dir / a.run
        records = [json.loads(line) for line in (run_dir / "circuit.jsonl").read_text().splitlines()]
        summary = summarize(records, a.attn_threshold)
        (run_dir / "circuit_summary.json").write_text(json.dumps(summary, indent=2))
        if not a.no_plot:
            from analysis.plots import plot_trajectory

            plot_trajectory(records, summary, a.run, run_dir / "trajectory.png")
    else:
        summary = analyze_run(a.run, a.results_dir, a.checkpoints_dir, a.max_docs, a.attn_threshold, not a.no_plot)
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
