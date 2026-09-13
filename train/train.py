"""Train a small HookedTransformer on a generated dataset, checkpointing every N steps.

    uv run python -m train.train --dataset datasets/base.npz --run base_s0

Writes:
- checkpoints/<run>/step_<N>.pt   (model config + state dict + training config)
- results/<run>/log.jsonl         (train loss and eval metrics over time)
- results/results.csv             (one row per finished run; circuit metrics left
                                   empty until analysis/ fills them in)
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

from data.generator import load_dataset

RESULTS_COLUMNS = [
    "run", "dataset", "condition", "seed", "steps", "tokens_seen",
    "train_loss", "val_loss", "val_io_acc", "val_io_logit_diff",
    "generalization", "generalization_pairs",
    "formation_step", "faithfulness", "sharpness", "zipper_score",
]
EVAL_SPLITS = ("val", "heldout_pairs", "heldout_io")
_EMPTY = {"tokens": np.zeros((0, 1), dtype=np.int64), "target_mask": np.zeros((0, 1), bool), "s_ids": np.zeros((0, 1), dtype=np.int64)}


@dataclass
class TrainConfig:
    dataset: str
    run: str
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    ckpt_every: int = 100
    eval_every: int = 100
    seed: int = 0
    device: str = "cpu"
    positional_embedding_type: str = "standard"  # or "rotary": relative positions, easy previous-token heads
    attn_only: bool = False  # no MLPs (the setting in which 2-layer induction heads form crisply)
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    switch_dataset: str = ""  # if set, train on this dataset from switch_step onward (data-schedule experiments)
    switch_step: int = 0


def build_model(tc: TrainConfig, d_vocab: int, n_ctx: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=tc.n_layers,
        d_model=tc.d_model,
        n_heads=tc.n_heads,
        d_head=tc.d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        act_fn="gelu",
        normalization_type="LN",
        seed=tc.seed,
        device=tc.device,
        positional_embedding_type=tc.positional_embedding_type,
        attn_only=tc.attn_only,
    )
    return HookedTransformer(cfg)


def lm_loss(logits: torch.Tensor, tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mean next-token cross-entropy, ignoring positions whose target is [PAD]."""
    pred = logits[:, :-1].reshape(-1, logits.shape[-1])
    tgt = tokens[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt, ignore_index=pad_id)


@torch.no_grad()
def evaluate(
    model: HookedTransformer, split: dict[str, np.ndarray], pad_id: int, batch_size: int = 256
) -> dict[str, float]:
    """Loss plus accuracy and IO-minus-S logit difference at indirect-object positions."""
    nan = float("nan")
    if len(split["tokens"]) == 0:
        return {"loss": nan, "io_acc": nan, "io_logit_diff": nan, "n_targets": 0}
    model.eval()
    losses, correct, diffs, n = [], 0, [], 0
    tokens_all = torch.as_tensor(split["tokens"])
    mask_all = torch.as_tensor(split["target_mask"])
    s_all = torch.as_tensor(split["s_ids"])
    for i in range(0, len(tokens_all), batch_size):
        tokens = tokens_all[i : i + batch_size].to(model.cfg.device)
        mask = mask_all[i : i + batch_size].to(model.cfg.device)
        s_ids = s_all[i : i + batch_size].to(model.cfg.device)
        logits = model(tokens)
        losses.append(lm_loss(logits, tokens, pad_id).item())
        # IO at position p is predicted from logits at p-1
        b, p = mask.nonzero(as_tuple=True)
        pred_logits = logits[b, p - 1]
        io = tokens[b, p]
        s = s_ids[b, p]
        correct += (pred_logits.argmax(-1) == io).sum().item()
        diffs.append(pred_logits[torch.arange(len(b)), io] - pred_logits[torch.arange(len(b)), s])
        n += len(b)
    model.train()
    diff = torch.cat(diffs) if diffs else torch.zeros(0)
    return {
        "loss": float(np.mean(losses)),
        "io_acc": correct / max(n, 1),
        "io_logit_diff": float(diff.mean()) if n else float("nan"),
        "n_targets": n,
    }


def save_checkpoint(model: HookedTransformer, tc: TrainConfig, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "train_config": asdict(tc),
            "model_cfg": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[HookedTransformer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["model_cfg"], device=device)
    model = HookedTransformer(HookedTransformerConfig.from_dict(cfg_dict))
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


class results_lock:
    """Exclusive file lock around results.csv edits; sweeps run jobs in parallel."""

    def __init__(self, path: Path) -> None:
        self.lock_path = path.with_suffix(".lock")

    def __enter__(self) -> Self:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.lock_path.open("w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def append_results_row(path: Path, row: dict[str, Any]) -> None:
    with results_lock(path):
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in RESULTS_COLUMNS})


def train(tc: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    splits, meta = load_dataset(Path(tc.dataset))
    pad_id = meta["pad_id"]
    train_tokens = torch.as_tensor(splits["train"]["tokens"])
    n_ctx = train_tokens.shape[1]

    model = build_model(tc, d_vocab=len(meta["vocab"]), n_ctx=n_ctx)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(tc.warmup_steps, 1))
    )

    ckpt_dir = Path(tc.checkpoints_dir) / tc.run
    run_dir = Path(tc.results_dir) / tc.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    (run_dir / "train_config.json").write_text(json.dumps(asdict(tc), indent=2))

    def log(record: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(f"run={tc.run} params={n_params:,} n_ctx={n_ctx} d_vocab={len(meta['vocab'])} steps={tc.steps}")
    save_checkpoint(model, tc, 0, ckpt_dir / "step_0.pt")
    t0 = time.time()
    loss_val = float("nan")
    latest: dict[str, Any] = {}
    switch_tokens = None
    if tc.switch_dataset:
        switch_splits, _ = load_dataset(Path(tc.switch_dataset))
        switch_tokens = torch.as_tensor(switch_splits["train"]["tokens"])
    for step in range(1, tc.steps + 1):
        source = switch_tokens if switch_tokens is not None and step > tc.switch_step else train_tokens
        idx = rng.integers(len(source), size=tc.batch_size)
        tokens = source[idx].to(tc.device)
        loss = lm_loss(model(tokens), tokens, pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_val = loss.item()

        if step % tc.eval_every == 0 or step == tc.steps:
            ev = {s: evaluate(model, splits.get(s, _EMPTY), pad_id) for s in EVAL_SPLITS}
            latest = {
                "step": step,
                "tokens_seen": step * tc.batch_size * n_ctx,
                "train_loss": loss_val,
                "elapsed_s": round(time.time() - t0, 1),
            }
            for s in EVAL_SPLITS:
                latest[f"{s}_loss"] = ev[s]["loss"]
                latest[f"{s}_io_acc"] = ev[s]["io_acc"]
                latest[f"{s}_io_logit_diff"] = ev[s]["io_logit_diff"]
            log(latest)
            print(
                f"step {step:6d} loss {loss_val:.3f} val {ev['val']['loss']:.3f} "
                f"io_acc {ev['val']['io_acc']:.3f} ld {ev['val']['io_logit_diff']:+.2f} | "
                f"heldout_io acc {ev['heldout_io']['io_acc']:.3f} ld {ev['heldout_io']['io_logit_diff']:+.2f} | "
                f"heldout_pairs acc {ev['heldout_pairs']['io_acc']:.3f} ({latest['elapsed_s']}s)"
            )
        if step % tc.ckpt_every == 0 or step == tc.steps:
            save_checkpoint(model, tc, step, ckpt_dir / f"step_{step}.pt")

    row = {
        "run": tc.run,
        "dataset": tc.dataset,
        "condition": json.dumps(meta["config"], sort_keys=True),
        "seed": tc.seed,
        "steps": tc.steps,
        "tokens_seen": latest.get("tokens_seen"),
        "train_loss": latest.get("train_loss"),
        "val_loss": latest.get("val_loss"),
        "val_io_acc": latest.get("val_io_acc"),
        "val_io_logit_diff": latest.get("val_io_logit_diff"),
        "generalization": latest.get("heldout_io_io_acc"),
        "generalization_pairs": latest.get("heldout_pairs_io_acc"),
        # Filled in by analysis/ once it exists:
        "formation_step": None,
        "faithfulness": None,
        "sharpness": None,
        "zipper_score": None,
    }
    append_results_row(Path(tc.results_dir) / "results.csv", row)
    print(f"checkpoints in {ckpt_dir}, log in {log_path}")
    return row


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", required=True)
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name in ("dataset", "run"):
            continue
        if isinstance(f.default, bool):
            p.add_argument(f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    return train(TrainConfig(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
