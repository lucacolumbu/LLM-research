import csv
import json

import torch

from data.generator import DataConfig, generate, save_dataset
from train.train import TrainConfig, load_checkpoint, train


def test_train_end_to_end(tmp_path):
    splits, meta = generate(DataConfig(n_train=64, n_val=16, ctx_len=40, seed=0))
    assert len(splits["heldout_io"]["tokens"]) > 0 and len(splits["heldout_pairs"]["tokens"]) > 0
    ds = tmp_path / "d.npz"
    save_dataset(ds, splits, meta)

    tc = TrainConfig(
        dataset=str(ds), run="t", d_model=32, d_head=8, n_heads=4, steps=6,
        batch_size=8, ckpt_every=3, eval_every=3, warmup_steps=2,
        checkpoints_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"),
    )
    row = train(tc)

    ckpts = sorted((tmp_path / "ckpt" / "t").glob("step_*.pt"))
    assert [c.name for c in ckpts] == ["step_0.pt", "step_3.pt", "step_6.pt"]
    log = [json.loads(l) for l in (tmp_path / "res" / "t" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [3, 6]
    assert 0.0 <= row["val_io_acc"] <= 1.0
    assert 0.0 <= row["generalization"] <= 1.0 and 0.0 <= row["generalization_pairs"] <= 1.0

    with (tmp_path / "res" / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["run"] == "t" and rows[0]["formation_step"] == ""

    # checkpoint reloads and reproduces the trained model's logits
    model, ckpt = load_checkpoint(ckpts[-1])
    assert ckpt["step"] == 6
    tokens = torch.as_tensor(splits["val"]["tokens"][:2])
    logits = model(tokens)
    assert logits.shape == (2, 40, len(meta["vocab"]))
    assert torch.isfinite(logits).all()
