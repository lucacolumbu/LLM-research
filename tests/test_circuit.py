import json

from analysis.circuit import analyze_checkpoint, analyze_run
from data.generator import DataConfig, generate, save_dataset
from train.train import TrainConfig, build_model, train


def test_decomposition_is_exact():
    splits, meta = generate(DataConfig(n_train=8, n_val=32, ctx_len=40, seed=3))
    tc = TrainConfig(dataset="", run="x", d_model=32, d_head=8, n_heads=4, seed=3)
    model = build_model(tc, d_vocab=len(meta["vocab"]), n_ctx=40)
    out = analyze_checkpoint(model, splits["val"], max_docs=32, batch_size=16)
    assert out["n_targets"] > 0
    assert abs(out["decomposed_ld"] - out["logit_diff"]) < 1e-3
    for k in ("attn_io", "attn_s", "dla_ld", "dla_io"):
        assert len(out[k]) == 2 and len(out[k][0]) == 4
    assert all(0.0 <= v <= 1.0 + 1e-6 for row in out["attn_io"] for v in row)
    empty = {k: v[:0] for k, v in splits["val"].items()}
    assert analyze_checkpoint(model, empty) == {}


def test_analyze_run_writes_outputs(tmp_path):
    splits, meta = generate(DataConfig(n_train=64, n_val=16, ctx_len=40, seed=0))
    ds = tmp_path / "d.npz"
    save_dataset(ds, splits, meta)
    tc = TrainConfig(
        dataset=str(ds), run="t", d_model=32, d_head=8, n_heads=4, steps=4, batch_size=8,
        ckpt_every=2, eval_every=2, warmup_steps=2,
        checkpoints_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"),
    )
    train(tc)
    summary = analyze_run("t", tmp_path / "res", tmp_path / "ckpt", max_docs=16)
    assert summary["steps"] == [0, 2, 4]
    assert len(summary["name_mover_candidates"]) == 2
    assert (tmp_path / "res" / "t" / "trajectory.png").stat().st_size > 10_000
    records = [json.loads(l) for l in (tmp_path / "res" / "t" / "circuit.jsonl").read_text().splitlines()]
    assert len(records) == 3 and set(records[0]) == {"step", "val", "heldout_pairs", "heldout_io"}
