import numpy as np
import torch

from analysis.patching import P, Q, analyze_prompts, analyze_run, build_prompts
from data.generator import VOCAB, DataConfig, generate, save_dataset
from train.train import TrainConfig, build_model, train


def test_prompts_are_well_formed():
    _, meta = generate(DataConfig(n_train=4, n_val=4, seed=0))
    pr = build_prompts(meta, 32, seed=1)
    assert pr["tokens"].shape == (32, 17)
    assert (pr["tokens"][:, Q] == VOCAB.id["to"]).all()
    assert (pr["tokens"][:, P] == pr["io"]).all() and (pr["tokens"][:, 10] == pr["s"]).all()
    assert (pr["corrupt"][:, 10] == pr["io"]).all()
    assert (np.delete(pr["corrupt"], 10, axis=1) == np.delete(pr["tokens"], 10, axis=1)).all()
    held = set(meta["heldout_io_name_ids"])
    assert not held & set(pr["io"].tolist()) and not held & set(pr["s"].tolist())
    hp = build_prompts(meta, 16, split="heldout_io")
    assert set(hp["io"].tolist()) <= held and not held & set(hp["s"].tolist())


def test_analyze_prompts_consistency():
    _, meta = generate(DataConfig(n_train=4, n_val=4, seed=0))
    model = build_model(TrainConfig(dataset="", run="x", d_model=32, d_head=8, n_heads=4, seed=1), len(meta["vocab"]), 17)
    pr = build_prompts(meta, 16, seed=2)
    out = analyze_prompts(model, pr, pool_ids=meta["pool_name_ids"])
    assert np.array(out["recovery_per_head"]).shape == (2, 4)
    curve = out["keep_only_curve"]
    assert curve[0]["k"] == 0 and curve[-1]["k"] == 8
    # keeping every head is the full model
    assert abs(curve[-1]["ld"] - out["ld_clean"]) < 1e-4
    assert "ctx_bonus" in out and "components_ctx" in out
    assert torch.isfinite(torch.tensor(out["circuit_recovery"]))


def test_analyze_run_writes_summary(tmp_path):
    splits, meta = generate(DataConfig(n_train=64, n_val=16, ctx_len=40, seed=0))
    ds = tmp_path / "d.npz"
    save_dataset(ds, splits, meta)
    tc = TrainConfig(dataset=str(ds), run="t", d_model=32, d_head=8, n_heads=4, steps=4, batch_size=8,
                     ckpt_every=2, eval_every=2, warmup_steps=2,
                     checkpoints_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"))
    train(tc)
    s = analyze_run("t", tmp_path / "res", tmp_path / "ckpt", n_prompts=16, update_results=True)
    assert len(s["circuit_heads"]) == 2 and (tmp_path / "res" / "t" / "patching.jsonl").exists()
    import pandas as pd
    df = pd.read_csv(tmp_path / "res" / "results.csv")
    assert {"formation_step", "faithfulness", "sharpness"} <= set(df.columns)
    assert df.loc[0, "sharpness"] == df.loc[0, "sharpness"]  # keep-only curve always yields a k
