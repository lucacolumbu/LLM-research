import numpy as np
import pytest

from analysis.zipper import Zipper, dataset_zipper_score, strip_pad
from data.generator import DataConfig, generate


@pytest.mark.parametrize("backend", ["zlib", "python"])
def test_repetition_compresses(backend):
    rng = np.random.default_rng(0)
    z = Zipper(backend, vocab_size=104)
    random_doc = rng.integers(2, 104, size=600)
    unit = rng.integers(2, 104, size=30)
    repeated = np.tile(unit, 20)
    assert z.length_bits(repeated) < 0.5 * z.length_bits(random_doc)
    assert z.compression_gain(repeated, rng) > 0.5
    assert abs(z.compression_gain(random_doc, rng)) < 0.15


@pytest.mark.parametrize("backend", ["zlib", "python"])
def test_conditional_score_drops_when_B_is_in_A(backend):
    rng = np.random.default_rng(1)
    z = Zipper(backend, vocab_size=104)
    B = rng.integers(2, 104, size=64)
    A_with = np.concatenate([rng.integers(2, 104, size=2000), B, rng.integers(2, 104, size=200)])
    A_without = rng.integers(2, 104, size=len(A_with))
    c_with, c_without = z.conditional_many([B], A_with)[0], z.conditional_many([B], A_without)[0]
    assert c_with < 0.5 * c_without
    assert 0.5 < c_without < 1.5


def test_dataset_score_tracks_repetition_knob():
    lo, _ = generate(DataConfig(n_train=300, n_val=10, repetition_rate=0.0, seed=0))
    hi, _ = generate(DataConfig(n_train=300, n_val=10, repetition_rate=1.0, seed=0))
    s_lo = dataset_zipper_score(lo, pad_id=0, n_docs=300)
    s_hi = dataset_zipper_score(hi, pad_id=0, n_docs=300)
    assert s_hi["mean_doc_gain"] > s_lo["mean_doc_gain"]
    assert s_hi["corpus_gain"] > s_lo["corpus_gain"]
    assert len(strip_pad(lo["train"]["tokens"][0], 0)) < 64
