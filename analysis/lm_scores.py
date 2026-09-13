"""Tier 2: LM cross-entropy scores.

Mean next-token loss of a document under a small model trained on the already-selected
set A (or any checkpoint from train/). Captures redundancy the zipper cannot see, but
only once the reference model has the relevant circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


@torch.no_grad()
def cross_entropy_per_doc(
    model: HookedTransformer, tokens: np.ndarray, pad_id: int, batch_size: int = 256
) -> np.ndarray:
    """Mean next-token cross-entropy of each document, ignoring [PAD] targets."""
    model.eval()
    out = []
    tokens_all = torch.as_tensor(np.asarray(tokens))
    for i in range(0, len(tokens_all), batch_size):
        tok = tokens_all[i : i + batch_size].to(model.cfg.device)
        logits = model(tok)
        tgt = tok[:, 1:]
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
        ).view(tgt.shape)
        keep = (tgt != pad_id).float()
        out.append(((ce * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu())
    model.train()
    return torch.cat(out).numpy()
