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

## 2026-09-12 (later): selection experiment, and the MLP claim reversed

### Phase 2, minimal: selecting 20k of a 60k heterogeneous pool

Pool: induction documents with per-document repeat fraction ~ U(0.1, 0.98). Subsets of
20,000 documents; attention-only rotary model, 6000 steps, 3 seeds. Table
`results/selection_summary.csv`.

| arm | selection rule | mean hidden fraction of selected docs | formation step (3 seeds) |
|---|---|---|---|
| random | uniform | 0.54 | 5200, never, never |
| zipper | top-N per-document LZ77 gain | 0.73 | 700, 700, 700 |
| refloss | lowest loss under a reference model trained on a random subset | 0.73 | 700, 700, 700 |
| oracle | top-N hidden fraction | 0.83 | 900, 1000, 900 |

Selecting by the zipper score forms the circuit at least 7 times sooner than a random
subset of the same size, with zero seed variance, and slightly sooner than the oracle
(the oracle picks the longest copies, which carry more ambiguous matches). Tier 2 matched
tier 1 almost document for document, but only because the reference model was the one
random-subset run that had formed an induction head; had it been either of the other two
seeds, reference loss would have carried no signal. That bootstrapping dependence is
exactly what the zipper avoids.

### The "MLPs block induction" claim was an optimisation artefact

With MLPs, rotary positions, r_max 31: learning rate 3e-4 (warmup 500) forms at
2250-2500; 1e-4 forms at 5500-6000; learning rate 1e-3 with weight decay 0 and warmup
500 forms at 1000-1250. The earlier failures used weight decay 0.01 with warmup 100. So
MLPs do not prevent the circuit; weight decay at that learning rate does. The repetition
sweep is being rerun with MLPs (rotary, weight decay 0) to remove the architecture
confound between the two tasks.

Addendum: with MLPs and *standard* learned positions, zero weight decay is not enough
(0.61 accuracy at 6000 steps for r_max 31, 0.10 for r_max 16). The working recipe for
models with MLPs is rotary positions plus zero weight decay; attention-only models form
the circuit under either position encoding. Position encoding and MLPs interact: a
previous-token head is cheap under rotary and expensive under learned absolute positions,
and only when it is cheap does the attention circuit win the race against MLP
memorisation. The two-name task is being rerun with the same recipe (`match_pool16_s*`)
so both tasks share one architecture.

## 2026-09-13: two-name task under the matched architecture (rotary, MLPs, weight decay 0)

Three seeds, pool 16, 4000 steps; patching with per-position mean ablation and an adaptive
circuit (smallest head set recovering 80% of the patching effect).

| seed | circuit heads | recovery | ablate: val / held-out IO | faithfulness | sharpness | heldout-IO acc at 4000 | attention of circuit heads (to IO, to S) |
|---|---|---|---|---|---|---|---|
| 0 | L1H1 | 1.01 | 0.47 / 0.00 | 0.11 | 4 | 0.09 | (0.00, 0.96) |
| 1 | L1H1, L1H0 | 0.93 | 0.60 / 0.06 | 0.08 | 4 | 0.06 | (0.00, 0.94), (0.00, 0.41) |
| 2 | L1H0, L1H1 | 0.93 | 0.58 / 0.00 | 0.12 | 3 | 0.02 | (0.00, 0.99), (0.01, 0.91) |

1. **Same mechanism under the induction task's architecture.** S-inhibition heads in layer
   1, no attention to the IO, and the in-context margin carried by the MLPs (10-11 logits
   vs under 1 from the heads' direct path). The cross-task claim no longer rests on an
   architectural difference.
2. **Weight decay was eroding the prior.** With weight decay off, held-out IO accuracy stays
   at 0.02-0.09 through 4000 steps, against 0.20-0.70 with weight decay 0.01 and otherwise
   identical training. The slow crawl of the held-out logit difference reported earlier is
   therefore largely weight decay shrinking the MLP prior, not the circuit strengthening,
   which fits the "byproduct of formation" reading and the late-leak result: the prior
   only moves when something pushes on it (decay or leaked examples).
3. **Per-position mean ablation changes the sparsity picture.** With one mean vector per
   head over all positions, sharpness was 7 of 8 heads; with per-position means it is 3-4
   of 8, and ablating the 1-2 circuit heads drops validation accuracy to 0.47-0.60, close
   to the two-candidate chance level the exclusion story predicts. The earlier sharpness
   numbers were an artefact of off-distribution ablation.

## 2026-09-13: sharpness done properly (greedy sufficient sets)

Ranking heads by single-head patching and keeping the top k says nothing about layer-0
heads that are individually negligible but collectively necessary, so keep-only curves
were flat. Greedy forward selection (add the head that most restores the logit difference,
everything else mean-ablated per position) gives minimal sufficient sets.

| model | seeds | ablate circuit heads: val acc | greedy sharpness (heads for 90% ld, out of total) | typical sufficient set |
|---|---|---|---|---|
| 2L4H standard, wd 0.01 | 3 | 0.43-0.48 | 2, 3, 5 of 8 | one S-inhibition head + one or two layer-0 heads |
| 2L4H rotary, wd 0 (matched) | 3 | 0.47-0.60 | 3, 4, 3 of 8 | same |
| 4L8H standard, wd 0.01 | 3 | 0.77-0.93 | 9, 7, 8 of 32 | 2-3 layer-0 heads, 2-3 layer-1 S-heads, 1-2 layer-2 heads |

In the small models the circuit is a quarter to a half of the model; in the larger model
it is about a quarter of the heads, spread over three layers, and ablating the top S-heads
alone barely dents validation accuracy because several backup S-heads exist. The
faithfulness figures reported earlier (keep-only-k by single-head ranking) should be
replaced by these.

## 2026-09-13: repetition sweep with MLPs restored (rotary, weight decay 0)

Same sweep as the attention-only one, matched architecture. Table
`results/summary_repeat_frac.csv` (last written for the `mlp_` runs); figure
`results/zipper_vs_formation_mlp.png`.

| repeat_frac | zipper gain | final acc | formation (acc >= 0.5) | 0.2 -> 0.8 width |
|---|---|---|---|---|
| 0.10-0.35 | 0.012-0.068 | 0.02-0.06 | never in 6000 | n/a |
| 0.50 | 0.101 | 0.22 | never in 6000 (reaches 0.2 at 5800-6000) | n/a |
| 0.75 | 0.154 | 0.68 | 4367 (351) | plateau at 0.68, no 0.8 |
| 0.98 | 0.201 | 0.90 | 1267 (58) | 100 steps, all seeds |

The ordering by zipper gain is unchanged (Spearman -0.81 over the 6 formed runs; the
censored conditions are the lowest gains), so the tier-1 result does not depend on the
attention-only architecture. Two differences: with MLPs everything forms 2-3 times later
and only the densest condition shows the clean snap, and at 0.75 the model sits at 0.68
accuracy, a partial solution the attention-only model never showed. MLPs offer a competing
partial path that delays and softens the transition. Runs at 0.35 and 0.5 are being
extended to 12,000 steps to fill the censored cells.

Addendum (12,000-step extension): repeat_frac 0.5 forms at 7000, 7000, 6600 and
plateaus near 0.75; 0.35 still never forms (0.07-0.17 at 12,000). Full ordering under the
matched architecture: 0.98 -> 1267, 0.75 -> 4367, 0.5 -> 6867, 0.35 and below -> never.
Monotone in the zipper gain, rank correlation -1 on condition means. The censoring at low
gains is now a budget statement (12,000 steps, 60k documents), not a gap in the ordering.

## 2026-09-13: the induction lag is a data-dependent free parameter (why MLP models plateau)

Direct probing of the models that copy without the textbook geometry. For each target at
copy position dst+j (query dst+j-1, needed token at src+j), attention mass at need+o for
o = 0..24 and, for layer 0, at query-o.

| condition | model | layer-1 lag (mass) | layer-0 offset | accuracy ceiling predicted from lag / observed |
|---|---|---|---|---|
| repeat 0.35, 0.5, 0.75 | attention-only | 0 (0.78-0.94) | previous token | 1.00 / 0.94-0.98 |
| noise 0.25 | attention-only | 0 (0.55-0.61) | previous token | |
| repeat 0.98 | attention-only | 9 (0.62-0.85) | 9-10 back | 0.99 / 0.91 |
| vocabulary 16 | attention-only | 8 (0.40-0.72) | 8-10 back | 0.94 / 0.84 |
| repeat 0.98 | MLPs | 9-10 (0.64-0.83) | 9-11 back | 0.98 / 0.90 |
| repeat 0.75 | MLPs | 19 (0.56-0.61) | 19-20 back | 0.76 / 0.71 |
| repeat 0.5 (12k steps) | MLPs | 12 (0.47-0.54) | 12-13 back | 0.83 / 0.73 |

A lag-k induction circuit is a layer-0 head carrying the token k+1 positions back and a
layer-1 head that matches the current token against it and reads the token k positions
after the needed one; it copies correctly whenever the copy offset exceeds k. On targets
with offset > lag every model scores 0.88-0.93; on targets with offset <= lag, 0.01-0.07.
The MLP "plateaus" are not partial circuits: they are complete lagged induction circuits
whose lag is longer than the copy offset in a quarter of the documents. Longer lags appear
with long copies (0.98), with small vocabularies, and with MLPs in every condition.

Consequences. (1) "Formation by attention to the copied token" (lag 0) undercounts: it
returned no formation for 0.98 and vocabulary 16, both of which formed lagged circuits.
Behavioural formation (accuracy >= 0.5) was used for all correlations, so the tier-1
results stand; the circuit analysis now needs a lag-aware measure. (2) "Cheapest circuit
the data allows" again: the same behaviour is delivered by a family of circuits indexed
by lag, and the data (and the presence of MLPs) selects the member, with an accuracy
ceiling as the price. Why longer lags are preferred under those conditions is open;
rotary distance resolution is one candidate.

## 2026-09-13: tier 2 sees what the zipper misses

Tier-2 dataset score: loss gain of validation documents under one fixed reference model
(the random-subset model from the selection experiment, which has an induction head),
relative to shuffled copies of the same documents (`analysis/tier2_dataset.py`). All 39
attention-only induction runs, four knob families. Figure `results/tiers_vs_formation.png`,
table `results/induction_tiers_vs_formation.csv`.

| score | Spearman vs formation, 30 formed runs | with 9 never-formed ranked last | repeat | noise | structure | vocabulary |
|---|---|---|---|---|---|---|
| zipper (tier 1) | -0.49 | -0.72 | -0.98 | -0.72 | -0.65 | -0.15 |
| reference-model loss (tier 2) | -0.85 | -0.93 | -0.98 | -0.93 | -0.52 | +0.09 |

Vocabulary axis, seed means: zipper 0.021 / 0.038 / 0.101 for 16 / 32 / 100 symbols;
tier 2 0.090 / 0.099 / 0.104; formation 1367 / 1167 / 1733. The reference model, whose
induction head is content-independent, scores the three vocabularies nearly alike, which
is what their formation times warrant; the zipper penalises small vocabularies because
accidental matches inflate its shuffled control. Pooled across families the tier-2 score
is the better predictor, entirely because of that family. Within every other family the
two tiers agree. This is the one-figure justification for the tiered framework: tier 1 is
free and right whenever the token distribution is fixed; tier 2 costs a trained reference
model and repairs the token-distribution blind spot.

### Lag-aware formation metric (re-run of all induction circuit summaries)

`analysis.circuit` now records, per head, the attention mass at need+o for o = 0..24 and
reports the dominant lag and the first checkpoint at which any head at any lag reaches
0.5. Across the 39 formed induction runs this attention-based formation step agrees with
the behavioural one (validation accuracy >= 0.5) with Spearman +0.99 and a mean absolute
gap of 87 steps, one checkpoint interval. The lag-0 metric returned nothing for repeat
0.98 (lag 9) and was unstable for vocabulary 16 (seeds split between lag 0 and lag 8).
Dominant lags by condition: 0 for attention-only models at repeat 0.35-0.75, noise 0.1 and
0.25, structure 2 and 4, vocabulary 32; 9 at repeat 0.98; 0 or 8 at vocabulary 16; 9-10,
19 and 12 for the MLP models at repeat 0.98, 0.75 and 0.5. The brief's formation
definition (patching recovery >= 50%) is still to be implemented for the induction task;
the two proxies now in use agree with each other.

## 2026-09-13: MLP repetition family completed (six formed conditions)

Matched architecture (MLPs, rotary, weight decay 0); 8,000 or 12,000 steps where needed.
Table `results/summary_repeat_frac.csv` (last written for the MLP family); figure
`results/zipper_vs_formation_mlp.png`.

| repeat_frac | zipper gain | formation (acc >= 0.5), 3 seeds | final acc | dominant lag (all 3 seeds) | r_max |
|---|---|---|---|---|---|
| 0.35 | 0.068 | never in 12,000 | 0.11 | (9, unformed) | 11 |
| 0.50 | 0.101 | 6867 (231) | 0.75 | 12 | 16 |
| 0.60 | 0.121 | 5467 (231) | 0.72 | 15 | 19 |
| 0.75 | 0.154 | 4367 (351) | 0.68 | 19 | 24 |
| 0.85 | 0.175 | 4133 (306) | 0.68 | 21 | 27 |
| 0.90 | 0.181 | 4000 (200) | 0.66 | 22 | 28 |
| 0.98 | 0.201 | 1267 (58) | 0.90 | 9-10 | 31 |

1. **The tier-1 ordering holds under the matched architecture with six points**: Spearman
   -0.92 over the 18 formed runs, -1.00 on condition means. Formation is 2-4 times slower
   than in attention-only models throughout.
2. **The lag is a deterministic function of the data.** All three seeds agree on the lag in
   every condition, and it grows with the maximum copy length: 12, 15, 19, 21, 22 for
   r_max 16, 19, 24, 27, 28, about 0.78 r_max. The accuracy ceiling falls accordingly
   (0.75 -> 0.66). At 0.98 the copy offset is nearly fixed by the geometry (source and
   copy each fill almost half the context), a different regime, and the lag drops to 9
   with a clean snap and a 0.90 ceiling.
3. Attention-only models choose lag 0 at every moderate fraction; the MLP models never do.
   Why MLPs and long copies favour long lags is the open mechanistic question in this
   dataset; it is the clearest example we have of the data (and the architecture) fixing a
   circuit parameter that the behaviour alone does not reveal.

## 2026-09-13: why the zipper beat the oracle in the selection experiment (item 5)

Lag-aware circuit summaries of all twelve selection-arm models: every formed model has a
lag-0 induction circuit (mass 0.82-0.87), so the arms differ in formation time, not
circuit variant. The difference is in what was selected. The oracle ranks documents by
the hidden per-document *maximum* copy length; the realised copy length is uniform below
it. The zipper ranks by realised verbatim content. Selected subsets:

| arm | mean hidden fraction | copied tokens per document | formation (3 training seeds) |
|---|---|---|---|
| zipper | 0.73 | 17.3 | 700, 700, 700 |
| oracle | 0.83 | 14.2 | 900, 1000, 900 |

The "oracle" was an oracle for the generator's parameter, not for the quantity that drives
formation. The zipper is closer to the true oracle (copied tokens). Both ran on one pool;
the replication on two further pools is queued.

## 2026-09-13: the brief's formation definition on the induction task (item 4)

`analysis/induction_patching.py`: corrupt the source segment, patch the layer-1 attention
outputs at target positions from the clean run, formation = first checkpoint with >= 50%
recovery of the clean-minus-corrupt log-probability, with a 1-nat effect floor (without
the floor the ratio is noise before the model can copy). Run on all 72 induction runs;
`results/induction_formation_measures.csv`.

| architecture | formed runs | Spearman, patching vs behavioural | patching minus behavioural (mean, mean abs, max abs) |
|---|---|---|---|
| attention-only | 27 | +0.99 | -26 / 70 / 300 steps |
| MLPs | 18 | +0.92 | -1083 / 1083 / 2000 steps |

Over all 45 formed runs: Spearman +0.96 against behavioural formation and +0.96 against the
lag-aware attention formation; final recovery 0.97-1.00 everywhere, so the layer-1 heads
carry the whole effect once formed. Two systematic differences: (1) in MLP models the
patched circuit reaches 50% recovery 1000-2000 steps before accuracy reaches 0.5, i.e. the
circuit is load-bearing while the behaviour is still climbing, which is the mechanistic
form of the "seep" observed in those models; (2) at noise 0.25 the clean-minus-corrupt
effect stays under the 1-nat floor (final accuracy 0.52-0.54), so patching never calls
formation although recovery is 1.0; the floor should scale with the attainable effect. For
the paper: the three definitions agree in rank; report patching as the primary metric and
note that it leads behaviour in the MLP regime.

## 2026-09-13: weight-decay control on the lag law (item 1)

MLP, rotary, warmup 500, 8000 steps, 3 seeds, weight decay 0.001 vs the 0 used for the law.

| repeat_frac | lag at wd 0 | lag at wd 0.001 | formation at wd 0 / 0.001 | final acc at wd 0 / 0.001 |
|---|---|---|---|---|
| 0.60 | 15 | 15, 15, 15 | 5467 / 5400 | 0.72 / 0.72 |
| 0.75 | 19 | 19, 19, 19 | 4367 / 4400 | 0.68 / 0.68 |
| 0.90 | 22 | 22, 22, 22 | 4000 / 4000 | 0.66 / 0.66 |

Lag, formation step and ceiling are unchanged. The lag is chosen by the data, not by the
regulariser. The wd 0.01 control is queued (that value blocked formation entirely under
learned positions and slowed it under rotary, so it tests the law at the edge of where the
circuit forms at all).

## 2026-09-13 (evening): batch 3 results

### Copy-length distribution at a fixed maximum (item 2): the lag follows neither the maximum nor the typical length

MLP, rotary, wd 0, repeat_frac 0.75 (maximum copy length 24), 8000 steps, 3 seeds.

| length distribution | mean / median copy length | zipper gain | formation | final acc | lag (3 seeds) |
|---|---|---|---|---|---|
| uniform in [2, 24] | 13.0 / 13 | 0.154 | 4000-4700 | 0.68 | 19, 19, 19 |
| short-tailed (80% in [2, 8]) | 6.6 / 6 | 0.069 | never in 8000 | 0.05 | unformed |
| fixed at 24 | 24 / 24 | 0.305 | 800, 800, 800 | 0.997 | 0, 0, 0 |

With every copy 24 tokens long the MLP model builds the textbook lag-0 circuit, snaps in at
800 and reaches 0.997; with the same maximum but uniform lengths it builds a lag-19 circuit
at 4000+ with a 0.68 ceiling; with mostly short copies nothing forms. So the lag is not a
function of the maximum copy length, and the "0.78 r_max" law from the repeat sweep was a
coincidence of the uniform distribution. The lag is set by the *distribution of copy lengths
and offsets* in a way that is still unexplained; note that lag 0 would cover every target
under uniform lengths, so the choice is not coverage-maximising. Zipper gain, which rises
sharply with copy length, orders the three conditions correctly (0.069 < 0.154 < 0.305 vs
never > 4300 > 800).

### Weight decay 0.01 (item 1, second value)
Lags 15 / 19 (one seed 18) / 22 at repeat 0.6 / 0.75 / 0.9, formation 5000-5400 / 4000-4600
/ 3800-4200, ceilings 0.71 / 0.69 / 0.66: indistinguishable from wd 0 and 0.001. Weight
decay does not touch the lag law. Correction to the earlier diagnosis: the blocker for MLP
models was learned absolute positions (and short warmup), not weight decay; the wd 0.01
rotary diagnostic that reached 0.72 was a lag ceiling, not a failure to form.

### Architecture robustness (item 7): 3 layers, 8 heads, attention-only
Formation 2600 (173) / 1300 (0) / 933 (153) for repeat 0.5 / 0.75 / 0.98; 0.35 never in
6000 (2L4H formed at 3600). Spearman -0.94 over 9 formed runs; jumps of 0.5-0.66 accuracy
per 100 steps. The headline correlation holds at a second depth and width.

### Selection replication on two new pools (item 5)
Pool 1: zipper 700, oracle 900, random never. Pool 2: zipper 700, oracle 800, random 6000.
With the original pool: zipper 700 x3, oracle 900-1000, random 5200/never/never. The zipper
subset forms at step 700 in every pool and seed; the parameter oracle is 100-300 steps
slower everywhere; random is 7x slower or fails.

### Phase 3 first attempt: failed for two reasons (item 6)
Five of six TinyStories runs died in `torch.save` with the disk full. The one that finished
(zipper subset, 8,000 documents of 128 tokens, 4,000 steps) shows the design flaw: the
subset is 1M tokens, so 4,000 steps is 32 epochs; train loss 1.12 vs validation 3.33, and
no induction head under either the uniform-token or the frequent-token prefix-matching
probe (score 0.02). A real-text selection experiment needs a pool at least an order of
magnitude larger (a slice of the TinyStories training split, ~400k stories from a 300 MB
range request) so that the selected subset supports multi-thousand-step training at low
epoch counts; it also needs disk space. Redesign pending.

## 2026-09-14: the zipper beats the dial-oracle because it measures realised copy length

Check on the original pool (`datasets/pool_het.npz`, 60k documents, one copied segment
each; the dial sets the maximum copy length, the realised length is uniform below it).

- Spearman of zipper gain with realised copy length: +0.995. With the dial: +0.60, which is
  also the correlation between the dial and copy length. The zipper is a near-perfect
  oracle for copy length; the dial is not.
- At matched dial settings the zipper's picks have longer copies than the oracle's: in the
  top bin (0.87-0.98) 20.5 vs 15.7 tokens (the oracle takes every document in the bin, so
  it inherits the pool mean); 18.7 vs 14.0 in the next; the zipper also reaches down to
  dial 0.3 to pick long-copy documents the oracle never sees.
- The oracle's subset is 24% documents with copies under 8 tokens; the zipper's has none.

Together with the copy-length distribution result (fixed long copies give the lag-0
circuit and a snap at 800; uniform lengths give lag 19 and 4000+; short copies give
nothing), this collapses two findings into one: copy length is the quantity that
determines which circuit forms and when, and gzip gain measures copy length almost
exactly. Test in flight: a subset selected by realised copy length itself (`copylen` arm)
should match or beat the zipper.

Confirmation: the copy-length oracle arm (`sel_copylen`, top 20k by realised copied
tokens, 97% overlap with the zipper's subset) forms at step 700 in all three seeds with
final accuracy 0.94-0.95, identical to the zipper arm. The compressor and the true oracle
for copy length are the same selector on this pool; the dial-oracle is the one that is
wrong, because it ranks by a parameter one step removed from what the circuit learns from.

## 2026-09-14: Phase 3 on a 379k-story TinyStories slice: negative at this scale, and the bridge says why

Pool 379,112 documents of 128 word-level tokens (vocabulary 8,192); 100k selected by gzip
gain (top 26%) vs three random 100k subsets; 2L4H MLP rotary, 8,000 steps (65M tokens, ~5
epochs of a subset), one run at a time on the GPU. Figure `results/phase3.png`, summary
`results/phase3_summary.json`.

| arm | gzip gain | LZ77 matches: median / mean longest / tokens in matches >= 8 | repeated-token frac | repeated-bigram frac | prefix-matching score at 8000 | induction acc | val loss at 8000 |
|---|---|---|---|---|---|---|---|
| zipper | 0.073 | 2 / 4.6 / 0.35% | 0.509 | 0.177 | 0.01-0.02 | 0.00 | 1.958-1.966 |
| random | 0.040 | 2 / 3.8 / 0.11% | 0.467 | 0.124 | 0.01 | 0.00 | 1.894-1.898 |
| synthetic zipper subset (for scale) | 0.215 | 16 / 17.4 / 27% | 0.417 | 0.266 | formed at 700 | 0.95 | |

1. No induction head formed in any of the six runs; the prefix-matching score stays at its
   uniform-attention baseline throughout, for both arms. The experiment therefore does not
   discriminate at this scale.
2. The match-length histogram is the bridge and the explanation. Natural children's stories
   contain almost no long verbatim copies: median match 2 tokens, and even the selected
   subset has 0.35% of tokens inside matches of 8 or more, seventy times less than the
   synthetic subset that forms at step 700. The score's dynamic range on this corpus is
   0.04 to 0.07 against 0.01 to 0.31 on synthetic data. Selecting by it shifts bigram
   repeats from 12% to 18% and long matches by a few tenths of a percent; it cannot
   manufacture copy structure the corpus does not have.
3. Single-token repetition, the thing natural-text induction heads mostly exploit, is
   already 47% in random stories and is not what the compressor rewards (rho 0.64 with
   gain, vs 0.79 for bigram repeats). A Phase 3 score aligned with natural-text induction
   should count repeated bigrams (Aoyama et al. 2026 use exactly that statistic); gzip gain
   is a proxy for it only at rho 0.79.
4. The zipper subset has *higher* validation loss (1.96 vs 1.90): selecting the most
   repetitive stories narrows the distribution.
Diagnostic in flight: attention-only rotary on the random subset, to learn whether an
induction head can form on this corpus at this budget at all (65M tokens is far below the
token counts at which natural-text induction heads are usually reported).

### Diagnostic: no induction on this corpus at 65M tokens in any architecture
Attention-only rotary on the random subset, 8,000 steps: prefix-matching 0.011 at the end
(baseline 0.02 at initialisation), induction accuracy 0.002, loss still falling (2.03).
In-distribution signals across checkpoints of all three models (MLP zipper, MLP random,
attention-only random): the loss advantage of tokens that repeat an earlier token in the
story over first occurrences is 1.1-1.2 nats at step 400 and does not grow to 8,000 (the
unigram effect of frequent words, not copying); loss at positions 100-127 stays higher
than at positions 5-20 in every checkpoint (early-minus-late gap -1.4 -> -0.7 nats), so
no in-context benefit of the kind that marks induction emerges. Conclusion: the TinyStories
Phase 3 at this token budget cannot discriminate any selector, because the target circuit
does not form. Reported natural-text induction phase changes occur at 10^9-10^10 tokens,
15-150x this budget; at 132 ms per 8k-token step on the M1 Pro that is 11-110 hours per
run, so a natural-text Phase 3 belongs on a rented GPU, or on a corpus with real verbatim
structure (source code), where the score has range and the circuit forms early.

## 2026-09-15: the second zipper quantity, corpus diversity (test 1, no new training)

Mean pairwise normalized compression distance NCD(A,B) = (L(AB) - min(L_A, L_B)) /
max(L_A, L_B) over 3,000 random document pairs per corpus, zlib on token bytes, nothing
subtracted, documents all 64 tokens (`analysis.zipper.corpus_diversity`,
`analysis/diversity.py`). Pool-size sweep, 15 runs; table `results/diversity_name_pool_size.csv`.

| pool | mean NCD | per-doc gain | held-out IO acc | formation (S-attention / patching) |
|---|---|---|---|---|
| 4 | 0.631 | 0.239 | 0.02 | 1600 / 867 |
| 8 | 0.694 | 0.221 | 0.17 | 333 / 200 |
| 16 | 0.728 | 0.215 | 0.47 | 333 / 267 |
| 32 | 0.748 | 0.215 | 0.55 | 400 / 333 |
| 64 | 0.758 | 0.211 | 0.57 | 467 / 400 |

Spearman over 15 runs: NCD vs held-out IO accuracy +0.80; NCD vs formation +0.02
(attention) and +0.10 (patching); per-document gain vs held-out IO accuracy -0.83; gain vs
formation -0.03 / -0.09. Three of the four predictions hold (diversity rises with pool size,
tracks generalisation, does not track formation). The fourth does not: within one knob the
per-document gain is monotone in pool size too, so it predicts generalisation as well as
NCD does, with the opposite sign. Separating the two quantities needs knobs that move one
and not the other: name skew at fixed pool size (diversity down, within-document repetition
roughly unchanged) and repetition rate at fixed pool size (gain up, diversity roughly
unchanged). Both sweeps are running.

## 2026-09-15: diversity vs per-document gain, dissociated (45 two-name runs)

Two new sweeps at pool 16, 3 seeds, 4000 steps: name skew (Zipf exponent 0, 0.5, 1, 1.5, 2)
and repetition rate (0, 0.25, 0.5, 0.75, 1). Skew lowers mean pairwise NCD (0.728 -> 0.647)
and barely moves gain (0.215 -> 0.236); repetition rate raises gain (0.097 -> 0.337) and
leaves NCD within 0.67-0.73. Pooled with the pool-size sweep the two quantities are
decorrelated (Spearman -0.27). Tables `results/diversity_<knob>.csv`, pooled
`results/diversity_pooled.csv`, figure `results/diversity_pooled.png`.

| outcome | n | Spearman NCD / gain | partial (other held fixed) NCD / gain |
|---|---|---|---|
| formation (S-attention) | 44 | +0.01 / +0.07 | -0.04 / +0.03 |
| formation (patching) | 15 | +0.10 / -0.09 | +0.16 / 0.00 |
| held-out pairs accuracy | 42 | +0.55 / -0.25 | +0.57 / -0.20 |
| held-out IO accuracy | 45 | +0.25 / +0.07 | +0.29 / +0.09 |
| sustained crossing of held-out logit diff | 25 | +0.80 / -0.41 | +0.75 / +0.21 |

1. **Neither quantity predicts when the two-name circuit forms.** Formation is flat
   (333-467) across every knob except the extremes (pool 4: 1600; repetition rate 1.0:
   1500, where every later sentence repeats an earlier triple and an induction shortcut
   makes the IOI circuit unnecessary for two thirds of targets, the brief's "too clean"
   prediction).
2. **Diversity is the quantity tied to generalisation; gain contributes nothing once
   diversity is fixed.** Held-out-pair accuracy +0.57 partial, the crossing step of the
   held-out logit difference +0.75 partial; gain's partials are -0.20 and +0.21.
3. **Held-out IO accuracy is only weakly a diversity effect pooled (+0.29)** because name
   skew moves it the other way: at pool 16, skew 0 -> 2 lowers diversity yet raises
   held-out IO accuracy 0.36 -> 0.63 and shortens the crossing from 2600 to 267 steps,
   while held-out-pair accuracy falls 0.999 -> 0.929. Frequency skew weakens the
   "never an answer" prior on the held-out names faster than it hurts pair
   generalisation; the mechanism is untraced.
4. Read together with the induction results: per-document gain predicts *when* the copy
   circuit forms (induction task, rho -0.98); corpus diversity predicts *how well* the
   circuit generalises (two-name task); each is silent on the other's outcome. The two
   zipper quantities map onto the two outcomes of the brief.
