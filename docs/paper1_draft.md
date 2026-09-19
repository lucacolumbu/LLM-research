# Compression predicts circuit formation: mechanism-informed data selection in small transformers

Draft v0.1, 2026-09-13. Numbers marked [pending] await runs queued in results/queue_batch3.sh and queue_batch4.sh.

## Abstract (draft)

Choosing training data by the mechanism it induces requires a per-document score that
predicts when a circuit forms. We show that the LZ77 compression gain of a document, a
hand-coded induction head, predicts the training step at which induction heads form in
2-layer transformers trained on synthetic copy data (Spearman -0.98 over 12 runs across
repeat fractions; -0.92 over 18 runs with MLPs), tracks the effect of noise and of repeat
structure at fixed repeated fraction, and fails only when the token distribution changes
(vocabulary size), where a reference-model loss score repairs it (-0.72 to -0.93 pooled over
39 runs). Selecting 20,000 of 60,000 mixed documents by compression gain forms the circuit at
step 700 in every seed, against 5,200 or never for a random subset. Two negative results
bound the method: the same behaviour is delivered by a family of induction circuits indexed
by a copy lag, and the data and architecture pick the member (lag 1 in attention-only models
at moderate repetition, lags 8 to 22 with long copies, small vocabularies, or MLPs), each with
a predictable accuracy ceiling; and an IOI-style two-name task is solved by subject
suppression plus MLP-mediated promotion of in-context names with no name-mover head, so a
copy-based score cannot see it. On real data the match-length histogram of a corpus says whether the score can act: on children's stories it cannot and nothing forms; on Python source it can, and gain-selected subsets grow a copy head sooner and five times stronger than random subsets, at a cost to the distribution that falls on rare identifiers. Adding the second compressor quantity, a compression-distance diversity constraint, yields the strongest copy head of any arm while recovering a third of that cost; a half-and-half mix recovers three quarters.

## 1. Introduction
- Data selection scores documents against loss or surface statistics (DSIR, DoReMi,
  RHO-LOSS, perplexity pruning, dedup). We score them against mechanism.
- Claim: for the induction head, a model-free compressor is such a score, with a known
  failure regime and a cheap repair.
- Method note the paper should state: each synthetic task was mechanism-traced before
  sweeping. Three shortcuts were caught this way (fixed-offset positional head, offset-r
  matching from fixed copy length, and the lag family itself). Data-selection work
  routinely skips this step.

## 2. Setup
- Tasks: (i) induction: [BOS] + random tokens with an embedded verbatim repeat of random
  length and offset; knobs repeat fraction, noise, vocabulary, repeat structure, copy-length
  distribution. (ii) Two-name IOI-style task with held-out-role generalisation splits.
- Models: 2 layers, 4 heads, d_model 128; attention-only rotary (main induction sweeps);
  MLPs + rotary + weight decay 0 (matched architecture for both tasks); 4L8H replication
  for the two-name task; 3L8H attention-only replication for the headline [pending].
- Formation: behavioural (validation accuracy on copied tokens >= 0.5), attention (lag-aware
  mass >= 0.5), patching (layer-1 patching recovers >= 50% of the source-corruption effect,
  1-nat floor). Across 45 formed runs the three agree in rank (Spearman 0.96); in
  attention-only models patching and behaviour coincide (mean gap 70 steps, max 300); in MLP
  models the patched circuit reaches 50% recovery 1000-2000 steps before accuracy reaches
  0.5, the mechanistic form of the slower "seep" in those models.
- Scores: tier 1, per-document LZ77 gain against a shuffled copy; tier 2, loss gain under a
  fixed reference model with an induction head.

## 3. Results

### 3.1 Compression gain predicts induction formation (pillar 1)
- Repeat fraction, attention-only: formation 3600 / 1733 / 900 / 800 for gains 0.068 / 0.101 /
  0.154 / 0.201; never within 6000 for 0.012 and 0.034. rho -0.98, seed spread 0-200 steps.
  Every formation is a jump of >= 0.4 accuracy within one 100-step interval.
- With MLPs (matched architecture): six formed conditions, rho -0.92 per run, -1.00 on
  condition means; formation 2-4x later; only the densest condition snaps.
- 3-layer 8-head attention-only: rho -0.94 over 9 formed runs, jumps intact.
- Noise: gain and formation fall together (rho -0.92). Structure at fixed repeated fraction:
  1, 2, 4 copies -> gain 0.154, 0.149, 0.125 -> formation 900, 1267, 1367 (rho -0.88):
  formation follows the score, not the fraction.
- Failure: vocabulary 16 vs 100 -> gain 0.021 vs 0.101, formation 1367 vs 1733 (wrong sign).
- Weight-decay control: at wd 0.001 the lags (15, 19, 22), formation steps and ceilings are
  unchanged from wd 0 in all seeds; the lag is set by the data. [wd 0.01 pending]

### 3.2 A reference model repairs the blind spot (pillar 2)
- Tier 2 scores vocabularies 16/32/100 as 0.090/0.099/0.104 where the zipper says
  0.021/0.038/0.101. Pooled over 39 runs and four families: zipper rho -0.72, tier 2 -0.93;
  identical within every family except vocabulary. Figure: results/tiers_vs_formation.png.
- Tier 2 depends on its reference having the circuit: in the selection experiment it matched
  tier 1 document for document only because its reference was the one random seed that
  formed.

### 3.3 Selection by compression works (pillar 3)
- 60k-document pool with hidden per-document repeat fraction ~ U(0.1, 0.98); 20k subsets.
  Formation: zipper 700/700/700; reference loss 700/700/700; parameter oracle 900/1000/900;
  random 5200/never/never. All formed models have lag-0 circuits (mass 0.82-0.87).
- Zipper beats the parameter oracle in all three pools (700 vs 800-1000). Reason: the dial
  sets a maximum copy length and the realised length is drawn below it; gzip gain tracks
  the realised length (rho 0.995) while the dial tracks it only at rho 0.60. At matched dial
  settings the zipper's picks carry longer copies (20.5 vs 15.7 tokens in the top bin); the
  oracle's subset is 24% short-copy documents, the zipper's 0%. A subset chosen by realised
  copy length overlaps the zipper's by 97% and forms at 700 in all three seeds: the
  compressor and the true oracle are the same selector.
- This joins pillar 4: copy length is what decides which induction circuit forms and how
  fast (fixed long copies: lag 0, snap at 800; uniform lengths: lag 19, 4000+; short: none),
  and gzip gain measures copy length. The compressor is not a proxy for repetition; it
  measures the quantity that determines learning.

### 3.4 The data picks the circuit variant (pillar 4)
- Lag-k induction: layer-0 head carries the token k+1 back, layer-1 head matches on it and
  reads k after the needed token. Complete copy mechanism with ceiling = fraction of targets
  with copy offset > k. Measured ceilings match: on targets with offset > lag every model
  scores 0.88-0.93; on the rest 0.01-0.07.
- Lags: 0 at repeat 0.35-0.75, noise, structure, vocabulary 32 (attention-only); 9 at repeat
  0.98; 0 or 8 at vocabulary 16; MLP family 12, 15, 19, 21, 22 for r_max 16, 19, 24, 27, 28
  9-10 at 0.98, all seeds agreeing. Not a function of the maximum: at a fixed maximum of 24,
  fixed-length copies give lag 0 (formation 800, acc 0.997), uniform lengths give lag 19
  (4000+, ceiling 0.68), mostly-short copies never form. Distance hypothesis rejected.
  Weight decay 0, 0.001, 0.01: no effect on lag, formation or ceiling.
- MLP "plateaus" (0.66-0.76) are lag ceilings, not partial circuits.
- Mechanism of the lag. Before any induction head exists, MLP models build a positional
  copier at one offset: a head that attends d back and copies, correct exactly when the
  copy offset is d+1, placed at the mode of the copy-offset distribution (24 in every
  dataset here; accuracy spikes of 0.3-0.8 at offsets 23-25 with nothing elsewhere). The
  induction head grows on that seed and inherits its offset: when the seed is a layer-0
  head, the lag is the seed offset minus one (uniform 22 -> 20 -> lag 19; geometric 23-24
  -> lag 22-23); when the seed is a layer-1 head and layer 0 supplies previous-token heads,
  the result is the textbook lag-0 circuit (fixed length). The lag is therefore set by the
  geometry of where copies sit relative to their sources, not by how much is copied; the
  copy-length distribution enters through the offset mode and the mass around it. Weight
  decay is irrelevant (0, 0.001, 0.01 identical). Open: what decides the seed's layer.
- Two-name task: S-inhibition heads (attention 0.94-0.99 on the subject, ~0 on the IO), IO
  margin of 10-11 logits over absent names carried by MLPs; ablating the circuit heads gives
  0.43-0.60 (chance for two candidates); held-out-IO failure carried by MLP 0 (-3 to -6
  logits), unembed bias ~0. Greedy sufficient sets 2-5 of 8 heads (2L4H), 7-9 of 32 (4L8H).
- The prior is a byproduct of formation: it deepens to -4.5 during formation; leaked
  examples cost ~800 before formation and ~100 after; with weight decay off it never erodes.

### 3.4b The second zipper quantity: diversity
- Mean pairwise NCD over document pairs. Three two-name knobs (pool size, name skew,
  repetition rate; 45 runs) decorrelate diversity from per-document gain (rho -0.27).
- Neither predicts formation (|rho| <= 0.1). On clean single-sentence probes, diversity has
  a modest positive partial effect on held-out-pair generalisation (+0.46 accuracy, +0.44
  logit difference) and gain a modest negative one (-0.19, -0.37); neither predicts
  held-out-role generalisation (+0.19 / -0.21). Gain predicts when the copy circuit forms;
  diversity relates to how far the two-name circuit generalises across pairs.
- Held-out-role generalisation is governed by the repetition rate of training documents,
  non-monotonically (probe accuracy 0.00, 0.25, 0.40, 0.06, 0.05 for rates 0 to 1): the
  copy circuit trained by in-document repeats extends its promotion of in-context names to
  the held-out ones until, at high repetition, it replaces the IOI circuit. A two-circuit
  interaction on mixed data. Figure: results/diversity_pooled.png.
- Two-name formation is flat except at the extremes: pool 4 (memorisation) and repetition
  rate 1.0, where an induction shortcut replaces the IOI circuit for most targets (1500 vs
  333), the brief's "too clean" prediction.

### 3.5 Real data: the diagnostic identifies structure, gain-only selection pays a cost, two quantities together are the remedy
- Diagnostic. The LZ77 match-length histogram of a corpus says whether the score can act.
  TinyStories (379k documents): median match 2 tokens, 0.1-0.35% of tokens in matches >= 8;
  no induction head forms in either arm at 65M tokens in any architecture. Python source
  (498k documents from 12k files): the top fifth by gzip gain has 45% of tokens in matches
  >= 8 and a mean longest match of 30 tokens, more than the synthetic subset that forms at
  step 700.
- Selection accelerates the target circuit, directionally and consistently. On code the
  gain-selected arm's prefix-matching score lifts from step 5,000 in all three seeds (0.03-
  0.07 at 65M tokens vs 0.01 random) and reaches 0.17 at 195M tokens, still rising, against
  0.03 for random. Formation (0.5) lies beyond the budget; reported natural-data phase
  changes sit at 1-10B tokens. The selected arm is further along at every checkpoint.
- The cost on the distribution is real and lands where copying should pay. Validation loss
  2.43 vs 2.03, and on second-and-later occurrences of rare identifiers, the positions an
  induction head exists for, the gain-only arm is 1.5-2 nats worse (6.2 vs 4.7): a gain-only
  selector picks repetitive boilerplate with fewer distinct identifiers to learn, the
  prediction the synthetic diversity results make about using one quantity alone.
- The remedy is the second compressor quantity. Greedy selection by gain with an NCD >=
  0.75 diversity constraint yields the strongest copy head of any arm (0.08, two seeds at
  0.10-0.12) and recovers a third of the loss (2.30; identifiers 5.68); a half-gain,
  half-random mix recovers three quarters (2.13; 5.13) with a head still twice random's.
  Gain buys the circuit, distance buys the distribution, and the constrained selector gets
  more of the first for less of the second than gain alone. This is the form of the method
  a code-model team could use: compress once, rank, dedup by compression distance.
- Open at this scale: no arm reaches formation, seed spread in the constrained arm is wide
  (0.02-0.12), and neither two-quantity arm closes the rare-identifier cost fully.

## 4. Limitations
- Toy scale, synthetic data; 2-layer models; one tokenizer per task.
- Stage dependence of any data score (Lee et al. 2025/26): the late-leak result is an
  instance; the selection experiment is single-stage.
- Persistence not tracked beyond 6-12k steps (Singh et al. 2023).
- The zipper measures within-document repetition; corpus-level duplication (Hernandez et al.
  2022) is a different quantity with the opposite effect.

## 4b. Future work (paper 2)
- Facts with paraphrases: a task whose circuit is compositional and whose training signal
  is semantic redundancy rather than verbatim repetition, where tier 1 should fail by
  construction and tier 2 or the circuit-probe learnability score (tier 3) has to take over.
- Interference between the copy circuit and the two-name circuit on mixed data; leak
  timing as a data-schedule lever.

## 5. Related work
See docs/literature-pass-2.md. Closest: Aoyama et al. 2026 (bigram statistics predict
emergence), Chen, Luo, Pan 2026 (mechanistic data attribution), Wang and Murfet 2026
(Patterning), ZIP-FIT, Sabry and Belz 2025/26, Akyurek et al. 2024 (n-gram heads), Wang and
Sato 2025 (position encoding and previous-token heads), Adhikari 2026 (attention-only IOI),
McDougall et al. 2023 (copy suppression).

## Figures
1. results/zipper_vs_formation_all.png (pillar 1 with noise, structure, vocabulary overlaid)
2. results/tiers_vs_formation.png (pillar 2)
3. selection: bar/strip of formation per arm (to make)
4. lag family: lag vs r_max, and ceiling predicted vs observed (to make)
5. two-name: trajectory with DLA decomposition; late-leak curve (results/leak_sample_efficiency.png)
