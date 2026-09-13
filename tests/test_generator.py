import numpy as np
import pytest

from data.generator import SPLITS, VOCAB, DataConfig, generate, load_dataset, save_dataset


def _cfg(**kw):
    base = {"n_train": 200, "n_val": 50, "ctx_len": 64, "seed": 1}
    base.update(kw)
    return DataConfig(**base)


def _pairs(tokens):
    """Unordered (A, B) name pairs, read off the 'when A and B' template."""
    when = VOCAB.id["when"]
    b, p = (tokens == when).nonzero()
    return [frozenset((int(tokens[i, j + 1]), int(tokens[i, j + 3]))) for i, j in zip(b, p)]


def _io_and_s(split):
    b, p = split["target_mask"].nonzero()
    return split["tokens"][b, p], split["s_ids"][b, p]


def test_shapes_and_targets():
    splits, _ = generate(_cfg())
    assert set(splits) == set(SPLITS)
    tok, mask, s_ids = (splits["train"][k] for k in ("tokens", "target_mask", "s_ids"))
    assert tok.shape == (200, 64) and mask.shape == tok.shape
    assert (tok[:, 0] == VOCAB.bos_id).all()
    assert np.isin(tok[mask], VOCAB.name_ids).all()
    b, p = mask.nonzero()
    assert (tok[b, p - 1] == VOCAB.id["to"]).all()
    assert (s_ids[mask] >= 0).all() and (s_ids[~mask] == -1).all()


def test_vocab_fixed_across_conditions():
    _, m1 = generate(_cfg(name_pool_size=4))
    _, m2 = generate(_cfg(name_pool_size=40, distractor_ratio=1.0))
    assert m1["vocab"] == m2["vocab"]


def test_heldout_io_names_are_trained_but_never_io():
    splits, meta = generate(_cfg(n_heldout_io_names=2))
    held = set(meta["heldout_io_name_ids"])
    assert len(held) == 2
    train_io, train_s = _io_and_s(splits["train"])
    assert not held & set(train_io.tolist()), "held-out name used as IO in training"
    # they do occur in training, and specifically as the predicted subject token
    assert held <= set(np.unique(splits["train"]["tokens"]).tolist())
    assert held <= set(train_s.tolist())
    test_io, test_s = _io_and_s(splits["heldout_io"])
    assert set(test_io.tolist()) <= held
    assert not held & set(test_s.tolist())


def test_heldout_pairs_never_cooccur_in_train():
    splits, meta = generate(_cfg(heldout_pair_frac=0.3))
    held = {frozenset(p) for p in meta["heldout_pairs"]}
    assert held
    assert not held & set(_pairs(splits["train"]["tokens"]))
    assert not held & set(_pairs(splits["val"]["tokens"]))
    assert set(_pairs(splits["heldout_pairs"]["tokens"])) <= held


def test_zero_holdouts_give_empty_splits():
    splits, meta = generate(_cfg(n_heldout_io_names=0, heldout_pair_frac=0.0))
    assert splits["heldout_io"]["tokens"].shape == (0, 64)
    assert splits["heldout_pairs"]["tokens"].shape == (0, 64)
    assert meta["heldout_io_name_ids"] == [] and meta["heldout_pairs"] == []


def test_invalid_configs():
    with pytest.raises(ValueError):
        _cfg(name_pool_size=4, n_heldout_io_names=3)
    with pytest.raises(ValueError):
        _cfg(heldout_pair_frac=1.0)


def test_knobs_move_stats():
    _, lo = generate(_cfg(repetition_rate=0.0))
    _, hi = generate(_cfg(repetition_rate=1.0))
    assert lo["stats"]["train"]["frac_repeated_sentences"] == 0.0
    assert hi["stats"]["train"]["frac_repeated_sentences"] > 0.5

    _, plain = generate(_cfg(distractor_ratio=0.0))
    _, filled = generate(_cfg(distractor_ratio=1.0))
    assert filled["stats"]["train"]["sentences_per_doc"] < plain["stats"]["train"]["sentences_per_doc"]

    _, uniform = generate(_cfg(name_pool_size=16, name_zipf=0.0))
    _, skewed = generate(_cfg(name_pool_size=16, name_zipf=2.0))
    assert uniform["name_entropy_bits"] == 4.0
    assert skewed["name_entropy_bits"] < 2.5


def test_target_noise_breaks_determinism():
    splits, meta = generate(_cfg(target_noise=1.0, name_pool_size=48))
    io, s = _io_and_s(splits["train"])
    assert (io == s).mean() > 0.005
    # noise never leaks a held-out name into the IO slot
    assert not set(meta["heldout_io_name_ids"]) & set(io.tolist())


def test_roundtrip(tmp_path):
    splits, meta = generate(_cfg())
    path = tmp_path / "d.npz"
    save_dataset(path, splits, meta)
    s2, m2 = load_dataset(path)
    assert m2 == meta
    for split in splits:
        for k in splits[split]:
            assert np.array_equal(splits[split][k], s2[split][k])


def test_deterministic():
    a, _ = generate(_cfg(seed=7))
    b, _ = generate(_cfg(seed=7))
    assert np.array_equal(a["train"]["tokens"], b["train"]["tokens"])


def test_heldout_io_leak_is_soft():
    splits, meta = generate(_cfg(n_train=400, heldout_io_leak=0.1))
    held = set(meta["heldout_io_name_ids"])
    io, _ = _io_and_s(splits["train"])
    frac = np.isin(io, list(held)).mean()
    assert 0.005 < frac < 0.06, frac  # ~10% of the ~23% of sentences with a held-out name
    assert abs(meta["stats"]["train"]["frac_heldout_io_sentences"] - frac) < 1e-9
    strict, _ = generate(_cfg(n_train=400, heldout_io_leak=0.0))
    assert meta["stats"]["train"]["frac_heldout_io_sentences"] > 0
    io0, _ = _io_and_s(strict["train"])
    assert not held & set(io0.tolist())
