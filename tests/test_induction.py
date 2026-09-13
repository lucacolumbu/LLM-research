import numpy as np
import pytest

from data.induction import BOS_ID, MAX_VOCAB, InductionConfig, generate


def test_repeat_structure():
    cfg = InductionConfig(repeat_frac=0.5, vocab_size=16, n_train=20, n_val=5, ctx_len=33, seed=0)
    splits, meta = generate(cfg)
    tok = splits["train"]["tokens"]
    r = cfg.repeat_len
    assert r == 8 and tok.shape == (20, 33)
    assert (tok[:, 0] == BOS_ID).all()
    mask = splits["train"]["target_mask"]
    lengths = mask.sum(1) + 1
    assert lengths.min() >= 2 and lengths.max() <= r and len(set(lengths.tolist())) > 2
    offsets = set()
    for i in range(20):
        p = mask[i].nonzero()[0]
        dst, ri = p[0] - 1, len(p) + 1  # copy start and length
        copied = tok[i, dst : dst + ri]
        srcs = [s for s in range(dst - ri + 1) if (tok[i, s : s + ri] == copied).all()]
        assert srcs, "no verbatim source for the copy"
        offsets.add(dst - srcs[0])
    assert len(offsets) > 3, "source-to-copy offset must vary across documents"
    b, p = mask.nonzero()
    assert (splits["train"]["s_ids"][b, p] == tok[b, p - 1]).all()
    assert tok.max() < 2 + 16 and len(meta["vocab"]) == 2 + MAX_VOCAB


def test_noise_and_validation():
    clean, _ = generate(InductionConfig(noise=0.0, n_train=50, n_val=5, seed=1))
    noisy, _ = generate(InductionConfig(noise=0.5, n_train=50, n_val=5, seed=1))
    def frac_copy_positions_matching(splits):
        tok, mask = splits["train"]["tokens"], splits["train"]["target_mask"]
        hits = total = 0
        for i in range(len(tok)):
            p = mask[i].nonzero()[0]
            dst, r = p[0] - 1, len(p) + 1
            seg = tok[i, dst : dst + r]
            best = max(((tok[i, s : s + r] == seg).sum() for s in range(dst - r + 1)), default=0)
            hits += best
            total += r
        return hits / total

    assert frac_copy_positions_matching(clean) == 1.0
    assert 0.35 < frac_copy_positions_matching(noisy) < 0.75
    with pytest.raises(ValueError):
        InductionConfig(repeat_frac=0.02)
    with pytest.raises(ValueError):
        InductionConfig(repeat_frac=1.0, ctx_len=64)


def test_n_repeats_and_heterogeneous_pool():
    one, m1 = generate(InductionConfig(repeat_frac=0.75, n_repeats=1, n_train=300, n_val=5, seed=2))
    four, m4 = generate(InductionConfig(repeat_frac=0.75, n_repeats=4, n_train=300, n_val=5, seed=2))
    c1, c4 = m1["stats"]["train"]["copied_tokens_per_doc"], m4["stats"]["train"]["copied_tokens_per_doc"]
    assert 0.6 < c4 / c1 < 1.6, (c1, c4)  # roughly the same copied fraction, different structure
    # more, shorter target runs with n_repeats=4
    def runs_per_doc(mask):
        d = np.diff(mask.astype(int), axis=1)
        return (d == 1).sum(1).mean()
    assert runs_per_doc(four["train"]["target_mask"]) > 1.8 * runs_per_doc(one["train"]["target_mask"])

    het, _ = generate(InductionConfig(repeat_frac=0.98, repeat_frac_min=0.1, n_train=300, n_val=5, seed=3))
    f = het["train"]["doc_frac"]
    assert f.min() >= 0.1 and f.max() <= 0.98 and f.std() > 0.15
    assert np.corrcoef(f, het["train"]["doc_copied"])[0, 1] > 0.6
    with pytest.raises(ValueError):
        InductionConfig(repeat_frac=0.5, repeat_frac_min=0.6)


def test_length_distributions_share_the_maximum():
    def lengths(cfg):
        splits, _ = generate(cfg)
        m = splits["train"]["target_mask"]
        return m.sum(1) + 1
    base = {"repeat_frac": 0.75, "n_train": 400, "n_val": 5, "seed": 4}
    uni, short, long_ = (lengths(InductionConfig(length_dist=d, **base)) for d in ("uniform", "short_tail", "long"))
    assert uni.max() == short.max() == long_.max() == 24
    assert np.median(short) < 0.6 * np.median(uni)
    assert (long_ == 24).all()
    with pytest.raises(ValueError):
        InductionConfig(length_dist="weird")
