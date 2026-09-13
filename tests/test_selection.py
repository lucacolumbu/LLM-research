import numpy as np

from data.induction import InductionConfig, generate
from selection.select_by_score import select


def test_zipper_and_oracle_pick_repeat_rich_docs():
    splits, meta = generate(InductionConfig(repeat_frac=0.98, repeat_frac_min=0.1, n_train=400, n_val=10, seed=0))
    pool = splits["train"]
    for method in ("zipper", "oracle"):
        idx, _ = select(pool, meta, method, n=100)
        assert len(idx) == 100 and len(set(idx.tolist())) == 100
        assert pool["doc_frac"][idx].mean() > pool["doc_frac"].mean() + 0.2
    idx_r, _ = select(pool, meta, "random", n=100, seed=1)
    assert abs(pool["doc_frac"][idx_r].mean() - pool["doc_frac"].mean()) < 0.15
    _, z = select(pool, meta, "zipper", n=100)
    # doc_frac is only the cap on copy length; the realised copied-token count is the cleaner target
    assert np.corrcoef(z, pool["doc_frac"])[0, 1] > 0.5
    assert np.corrcoef(z, pool["doc_copied"])[0, 1] > 0.7
