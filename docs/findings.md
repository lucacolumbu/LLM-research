# Findings log

## 2026-09-12: name-pool-size sweep and soft-leak condition

Setup: 2-layer, 4-head, d_model 128 HookedTransformer; 4000 steps, batch 64, ctx 64;
checkpoints every 200; 3 seeds per condition; `repetition_rate=0.5`, 2 held-out IO names,
20% held-out pairs. Figures: `results/summary_name_pool_size.png`,
`results/summary_heldout_io_leak.png`, per-run `results/<run>/trajectory.png`.

| pool size | zipper | held-out IO acc (mean, std) | held-out pairs acc | S-attention formation step | sustained crossing step | heads DLA | MLP0 DLA |
|---|---|---|---|---|---|---|---|
| 4  | 0.239 | 0.02 (0.04) | n/a  | 1600 (unstable) | never (0/3) | -0.8 | +0.2 |
| 8  | 0.221 | 0.17 (0.18) | 0.96 | 333  | 2500 (2/3)  | +1.1 | -1.9 |
| 16 | 0.215 | 0.47 (0.25) | 1.00 | 333  | 3100 (2/3)  | +2.7 | -2.9 |
| 32 | 0.215 | 0.55 (0.19) | 1.00 | 400  | 3600 (3/3)  | +5.0 | -4.9 |
| 64 | 0.211 | 0.57 (0.14) | 1.00 | 467  | 3300 (2/3)  | +4.7 | -5.8 |

DLA = direct logit attribution to the IO-minus-S logit difference on the held-out IO
split at the final checkpoint. The LayerNorm-bias plus unembed-bias term is -0.04 to
-0.10 in every condition.

1. **The circuit is S-inhibition, not a name-mover.** In every run with pool >= 8 the two
   top-attribution heads sit in layer 1, put 0.9-1.0 of their attention on the subject
   name and push its logit down. No head puts more than 0.2 on the IO name at any
   checkpoint. The IO identity reaches the output through the MLPs.
2. **The held-out-IO metric is a difference of two terms that pool size moves together.**
   Larger pools strengthen the head term (+1 -> +5) and deepen the MLP0 suppression of
   the held-out names (-2 -> -6). Held-out IO accuracy is the residual, which is why it
   rises only from 0.17 to 0.57 while both terms triple.
3. **The prior lives in MLP0, not the output bias.** The unembed and LN biases are
   negligible everywhere.
4. **The crossing is a crawl, not a jump.** The held-out-IO logit difference climbs
   roughly linearly from about -6 at step 200 towards 0 over ~2500 steps, then hovers at
   0 +/- 2 with large checkpoint-to-checkpoint noise. No phase transition. Seed variance
   is large: at pool 16 the final accuracy is 0.20, 0.50 and 0.70 across seeds.
5. **Pool 4 is the memorisation regime.** Two regular names plus two held-out names:
   the head term is negative, MLP1 does the work, S-attention forms late if at all, and
   held-out IO accuracy stays at zero.
6. **A 1.2% leak removes the prior entirely.** With held-out names as IO in 1.16% of
   training sentences (~2 examples per step), all three seeds reach 100% held-out IO
   accuracy by step 400 (between roughly 450 and 900 leaked examples seen) and MLP0's
   term shrinks from -3 to -0.7. Sample efficiency needs finer early checkpoints to pin
   down.
7. **Tier-1 zipper score barely moves along this axis** (0.24 -> 0.21) because pool size
   changes name repetition only slightly; the zipper is expected to track
   `repetition_rate`, not pool entropy. It is recorded per run for the later correlation.

Caveats: single architecture and learning rate; formation step is the attention-based
proxy, not activation patching; crossing statistics are sensitive to the hovering noise
(the sustained-crossing rule needs 75% of later checkpoints positive).

Next: activation patching for the brief's formation and faithfulness definitions; a
`repetition_rate` sweep, which is the axis the zipper score should predict; finer early
checkpoints for the leak condition to measure copy sample efficiency.

## 2026-09-12 (later): patching closes the loop on the two-name task; leak sample efficiency

Method: 256 single-sentence prompts per run, corrupted twin with the subject slot swapped,
answer read at "to". Circuit heads = top-2 by single-head patching recovery at the final
checkpoint. Figures and per-run detail in `results/<run>/patching_summary.json`;
aggregate columns in `results/summary_name_pool_size.csv`.

| pool | formation (patching, >=50% recovery) | recovery, top-2 heads | val acc after ablating them | held-out IO acc after ablating them | faithfulness (only 2 heads kept) | sharpness (heads for 90% ld) | context bonus, val | context bonus, held-out IO |
|---|---|---|---|---|---|---|---|---|
| 4  | 200-1800 | 0.93-1.00 | 0.58-0.70 | n/a | 1.00 | 1 | 9.8 | n/a |
| 8  | 200 | 0.95 | 0.60 | 0.02 | 0.23 | 4 | 9.9 | 2.0 |
| 16 | 333 | 0.94 | 0.61 | 0.00 | 0.04 | 6.3 | 11.1 | 5.0 |
| 32 | 333 | 0.86 | 0.74 | 0.00 | 0.01 | 7.7 | 11.6 | 4.8 |
| 64 | 400 | 0.87 | 0.64 | 0.02 | 0.00 | 7.7 | 11.9 | 7.3 |

1. **The S-signal is carried by two layer-1 heads.** Patching only their outputs at the
   answer position recovers 86-100% of the clean-minus-corrupt logit difference in every
   run with pool >= 8. The brief's patching-based formation step (200-400) agrees with the
   S-attention proxy (333-467).
2. **Ablating them does not fully collapse validation.** Accuracy falls to 0.43-0.51 in
   five of twelve runs and only to 0.75-0.88 in the others, where a third layer-1 head
   backs them up. On held-out IO names ablation takes accuracy to 0.00 everywhere: for
   names the MLP prior opposes, S-suppression is necessary.
3. **The exclusion story is half right.** There is a large IO-specific signal: the IO
   logit exceeds the mean logit of pool names absent from the prompt by 10-12 logits, and
   that margin comes from MLP 1 (5-8 logits) and MLP 0 (about 2.5), with the heads' direct
   path contributing under 0.5. So the model does represent "which names are in the
   context", but through MLP features fed by layer-0 attention, not through a name-mover.
   The mechanism is: MLPs promote in-context names, layer-1 heads demote the subject.
   Neither half is attention copying, so the zipper cannot see it.
4. **The prior is partly in MLP 1 too.** The context bonus for held-out names is only 2-7
   logits versus 10-12 for regular names, and it grows with pool size (2.0 -> 7.3), which
   is the mechanism behind the generalisation trend: bigger pools make MLP 1's in-context
   promotion more token-general. In the 1.2%-leak runs the held-out bonus is 9-11, the
   same as for regular names.
5. **The circuit is not sparse at the head level.** Keeping only the two S-heads and
   mean-ablating the other six gives near-zero accuracy; reaching 90% of the logit
   difference needs 6-8 of 8 heads, because layer-0 heads feed both the S-detection and
   the MLP name features. In a 2-layer 4-head model the "circuit" is most of the model.
   Pool 4 is the exception: one head suffices (sharpness 1, faithfulness 1.0).

**Leak sample efficiency** (`results/leak_sample_efficiency.png`, checkpoints every 25
steps, ~2.2-2.6 leaked examples per step). The held-out logit difference first *deepens*
to about -4.5 over the first ~250 leaked examples, while the main circuit forms, then
climbs roughly linearly at ~0.01 logits per leaked example and crosses zero between 600
and 1000 examples (first checkpoint with accuracy >= 0.5 at 670, 983 and 659 examples for
the three seeds). Accuracy is a threshold of the linear climb, not a separate event. About
a thousand examples, roughly 1% of the sentences seen by then, overwrite the entry.

**Induction task** (`data/induction.py`, smoke run): accuracy jumps from 0.02 at step 100
to 1.00 at step 200. A snap, against the crawl of the two-name task. Repeat-fraction sweep
in progress.

## 2026-09-12 (later): the induction task had to be redesigned twice

Goal: a task where attention copying is the only viable mechanism, so the LZ77 hypothesis
can be tested on a repetition-rate sweep. Each design was checked by tracing the trained
model's mechanism before sweeping; each time the model found a cheaper mechanism than the
one the task was meant to force.

1. **Fixed offset.** First version: random tokens, the last r tokens copy the first r.
   Every document has the same source-to-copy offset. The model solved it with a single
   layer-0 head attending r positions back (attention 0.97), with no content matching.
   Accuracy jumped from 0.02 to 1.00 between steps 100 and 200.
2. **Random offset, fixed length.** Source and copy positions randomised, r fixed per
   condition. Genuine content-based matching appeared (corrupting the source token or its
   predecessor kills the prediction; corrupting anything else does not; killing either
   attention hop kills it). But the geometry is not the textbook circuit: layer-0 heads at
   key position k gather the tokens at k-r and k-r+1 (r = copy length), and the layer-1
   heads match the current token against that gathered token and read the gathered
   successor. It is an induction head whose "previous token" relation is dilated by the
   copy length, because with a fixed r "the token r back" is a privileged feature (when
   the copy immediately follows the source, it is the answer with no matching at all).
   Accuracy jumped from 0.20 to 0.81 in one 250-step interval; all four layer-1 heads
   became near-identical copies of the same head.
3. **Random offset, random length (current).** r ~ U{2..r_max} per document. No positional
   shortcut survives. In a 3000-step smoke run the model learned nothing (loss at the
   uniform floor), so the canonical circuit is markedly harder to reach than either
   shortcut. Diagnostics in progress: longer copies, higher learning rate, more steps.

Method lesson: for every synthetic task, trace the mechanism of one trained model with
edge knockouts before sweeping. Attention-to-position metrics presuppose a geometry; use
the behavioural formation step (first checkpoint with accuracy >= 0.9) or patching for
anything mechanism-agnostic.

### Induction diagnostics (variable offset and length, 60k documents, r_max 16 or 31)

| model | positions | r_max | accuracy trajectory |
|---|---|---|---|
| with MLPs | standard | 16 | 0.07 at 3000, 0.17 at 8000: crawl |
| with MLPs | standard | 31 | 0.20 at 4000: crawl |
| with MLPs, lr 3e-3 | standard | 16 | 0.08 at 4000 |
| with MLPs | rotary | 31 | 0.40 at 4000, plateau at 0.71 by 10000 |
| attention-only | rotary | 31 | 0.09 -> 0.51 -> 0.90 at steps 500, 750, 1000 |
| attention-only | rotary | 16 | 0.08 -> 0.90 between steps 1500 and 1750 |
| attention-only | standard | 31 | 0.13 -> 0.59 -> 0.80 at steps 750, 1000, 1250 |

The MLPs were the blocker, not the position encoding: with MLPs the model memorises the
training documents (validation loss rises) and the two-head circuit never gets its
phase change. Attention-only models snap in, as in the induction-heads literature. The
attention-only rotary model has the textbook geometry: layer-1 heads attend to the token
to copy with mass 0.9-1.0, layer-0 heads at that key attend one position back. The
induction sweep therefore uses attention-only rotary models; the two-name task keeps MLPs
because they carry its in-context name promotion. Cross-task comparisons must respect
that architectural difference.

## 2026-09-12 (later): tier-1 test on the induction task

Setup: attention-only 2-layer rotary model, 60k documents per condition, 6000 steps,
checkpoints every 100, 3 seeds. Knob: `repeat_frac` (maximum copy length as a fraction of
half the context; per-document length and offset random). Figure:
`results/zipper_vs_formation.png`; table `results/summary_repeat_frac.csv`.

| repeat_frac | zipper gain | final acc | formation (acc >= 0.5) | formation (top-head attention to copied token >= 0.5) | largest accuracy gain in one 100-step interval |
|---|---|---|---|---|---|
| 0.10 | 0.012 | 0.02 | never in 6000 | never | 0.01 |
| 0.20 | 0.034 | 0.04 | never in 6000 | never | 0.01 |
| 0.35 | 0.068 | 0.94 | 3600 (200) | 3600 | 0.45 |
| 0.50 | 0.101 | 0.97 | 1733 (58) | 1733 | 0.60 |
| 0.75 | 0.154 | 0.98 | 900 (0) | 900 | 0.74 |
| 0.98 | 0.201 | 0.90 | 800 (0) | n/a, attention split by ambiguous matches | 0.64 |

1. **The zipper gain predicts induction-head formation.** Spearman rho = -0.98 between
   per-document compression gain and formation step over the 12 runs that formed (the 6
   that never formed have the two lowest gains, consistent with the ranking). Seed
   variance is small: 0 to 200 steps. Formation time roughly halves each time the gain
   doubles (3600 -> 1733 -> 900 for gains 0.068 -> 0.101 -> 0.154).
2. **The copy circuit snaps in.** Accuracy goes from below 0.2 to above 0.8 within one or
   two 100-step intervals in every run that forms, and the attention-based formation step
   coincides with the behavioural one exactly. This is the predicted contrast with the
   two-name task, where the held-out-IO logit difference crawls across 2500 steps.
3. **Caveat on the test.** Along this axis the zipper gain is a deterministic function of
   the knob (seed std 0.001), so the test shows the zipper orders the *repetition* axis
   correctly; it does not yet show the zipper outperforms simply knowing the knob. The
   informative next test dissociates them: sweep `noise` (copied tokens randomly replaced)
   and `vocab_size`, which change formation while barely moving the compression gain, and
   see where the zipper fails, as the brief predicts it should for non-verbatim structure.
4. The accuracy ceiling at repeat_frac 0.98 (0.90) comes from ambiguous matches: with
   copies up to 31 tokens and a vocabulary of 100, the current token often has a spurious
   earlier occurrence. The attention-based formation step is undefined there because the
   top head's attention to the copied token stays below 0.5.

## 2026-09-12 (later): the prior is cheap to overwrite after formation, not before

Late-leak experiment: train on the no-leak pool-16 data until step 400 (validation
accuracy 1.0 from about step 325), then switch to the 1.2%-leak data. Checkpoints every
25 steps, 3 seeds. Compared with the leak-from-step-0 runs.

| condition | leaked examples until held-out IO accuracy >= 0.5 | logit difference just before / 100 steps after |
|---|---|---|
| leak from step 0 | 670, 983, 659 | n/a (starts at 0, dips to -4.5 first) |
| leak from step 400 | 112, 112, 56 | -5.1 -> +4.6, -4.4 -> +3.9, -4.7 -> +3.2 |

Six to twelve times fewer examples are needed once the circuit exists, and the flip is
stable (accuracy 0.99-1.00 for the remaining 1000 steps). This is the opposite of the
"cheapest to prevent during formation" prediction. Reading: the "never an answer" prior
is built by the same gradient that builds the circuit, so leaked examples that arrive
during formation are mostly wasted against it, while after formation only a token-level
entry in the MLP has to move and each example shifts it by roughly 0.1 logits. For data
design this says: to make a rare answer class available, spend the examples after the
mechanism is in place rather than mixing them in from the start.

## 2026-09-12 (later): dissociation A, noise and vocabulary (induction task)

Same setup as the repetition sweep, `repeat_frac` fixed at 0.5. Tables in
`results/summary_noise.csv` and `results/summary_vocab_size.csv`.

| knob | value | zipper gain | final acc | formation (acc >= 0.5) |
|---|---|---|---|---|
| noise | 0 | 0.101 | 0.97 | 1733 (58) |
| noise | 0.10 | 0.076 | 0.80 | 2233 (58) |
| noise | 0.25 | 0.047 | 0.54 | 3933 (153) |
| noise | 0.50 | 0.015 | 0.03 | never in 6000 |
| vocab | 16 | 0.021 | 0.87 | 1367 (462) |
| vocab | 32 | 0.038 | 0.92 | 1167 (58) |
| vocab | 100 | 0.101 | 0.97 | 1733 (58) |

1. **Noise is not a dissociation: the zipper sees it and is right.** Random replacement
   of copied tokens shortens verbatim matches, so the gain falls with noise, and formation
   slows in step with it (Spearman -0.92 over the 9 formed runs). At 50% noise neither the
   gain nor the circuit survives.
2. **Vocabulary is a dissociation, and the zipper gets it wrong.** Shrinking the vocabulary
   from 100 to 16 symbols cuts the gain fivefold, because the shuffled control sequence is
   itself full of accidental 2- and 3-token matches, yet the circuit forms *faster*
   (1367 vs 1733 steps; 1167 at 32 symbols). Spearman between gain and formation across the
   vocabulary axis is +0.27, the wrong sign. The zipper conflates learnable copy structure
   with coincidental repeats: it is a measure of verbatim redundancy relative to chance,
   and when chance repeats are common it under-reports the signal the circuit can use.
   Formation is faster at small vocabularies presumably because there are fewer token
   embeddings to learn and spurious matches give partial credit early.

Net: the zipper is a good predictor along axes that change the amount or integrity of
verbatim structure (repeat fraction, noise) and a poor one along axes that change the
token distribution (vocabulary entropy). That is the boundary the paper should state.

## 2026-09-12 (later): dissociation B, repeat structure at fixed repeated fraction

`repeat_frac` 0.75, `n_repeats` 1, 2 or 4 copied segments per document with the
per-segment cap divided accordingly, so the copied fraction stays at 0.187 of tokens
(copied tokens per document 14, 14, 16). Table `results/summary_n_repeats.csv`; figure
`results/zipper_vs_formation_all.png` (green squares: vocabulary; orange: noise; the
structure runs are the third group).

| repeats per doc | zipper gain | final acc | formation (acc >= 0.5) |
|---|---|---|---|
| 1 | 0.154 | 0.98 | 900 (0) |
| 2 | 0.149 | 0.97 | 1267 (58) |
| 4 | 0.125 | 0.94 | 1367 (58) |

Splitting the same amount of repetition into shorter copies lowers the LZ77 gain (each
match costs a fixed overhead, so short matches compress worse) and delays formation, even
though the number of copied tokens is equal or slightly higher. Formation tracks the
score, not the repeated fraction: Spearman -0.88 over 9 runs. This is the evidence that
the compression score carries information the generating knob does not, which is what a
reviewer will ask for. Combined with dissociation A: the zipper predicts formation along
every axis that changes the amount, integrity or granularity of verbatim structure, and
fails only where the token distribution changes (vocabulary size).
