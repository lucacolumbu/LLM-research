# Paper 1 notes

## Working claim

Per-document LZ77 compression gain (the "zipper" score) predicts when a copying circuit
forms in a small transformer, because LZ77 and an induction head exploit the same
verbatim repeated structure. On the pure induction task, formation step is a monotone
function of the score (Spearman -0.98 over 12 runs, seed spread under 200 steps), and the
circuit arrives as a phase transition. The same score cannot predict the two-name task,
whose circuit is exclusion plus MLP set membership rather than copying, and the score is
flat along that task's main data axis. Data controls *which* circuit forms, not only when.

## Structure of the evidence (planned)

1. Repetition sweep: score orders formation (done).
2. Dissociation A, knobs that move formation but not the score: `noise`, `vocab_size`
   (running). Bounds what the zipper can see.
3. Dissociation B, knobs that move the score but not the repeated fraction: `n_repeats`
   (few long vs many short copies) (running). Tests whether formation tracks the score or
   the fraction, i.e. whether compression measures the right thing.
4. Selection: heterogeneous pool, fixed-size subsets chosen by zipper, random, reference
   loss, oracle; compare formation (queued). The abstract's first sentence.
5. Negative control: the two-name task, where the mechanism is not copying and the score
   is flat (done).

## Methodological point to state explicitly

Validate the mechanism before sweeping the data. The induction task was redesigned three
times because tracing one trained model with edge knockouts found a shortcut each time:
a positional head at a fixed offset, an induction match built on "the token r back" when
the copy length was fixed, and a memorising MLP path that starved the attention circuit.
Each shortcut would have produced a clean-looking sweep about the wrong mechanism. Most
data-selection work scores data against loss and never checks which mechanism the loss
is buying; the mechanism check is the method, not a preliminary.

## Findings that need their own paragraph or figure

- Exclusion, not copying, on the two-name task: S-inhibition heads plus MLP promotion of
  in-context names, no name mover; backup-head redundancy in half the seeds.
- The "never an answer" prior is a byproduct of forming the circuit, not a pre-existing
  bias: it deepens during formation. Overwriting it afterwards is cheap (late-leak
  experiment, pending confirmation).
- The apparent "MLPs block induction" result was an optimisation artefact (weight decay
  0.01 with learned positions); with rotary positions and weight decay 0 the MLP model
  forms the circuit by ~1200 steps. Both tasks now run under one architecture.
- Weight decay is what erodes the two-name prior: with it off, held-out IO accuracy
  stays near zero for 4000 steps; the prior moves only when pushed (decay or leaks).
- The prior is cheap to overwrite after formation (~100 leaked examples) and expensive
  to prevent during it (~800): spend rare-class examples after the mechanism exists.

## Caveats to state

- Sparsity: greedy sufficient sets are 2-5 of 8 heads at 2L4H and 7-9 of 32 at 4L8H.
  Single-head-ranked keep-only curves are not a valid sparsity measure here.
- The original sweeps used different architectures per task; the matched replications
  (`mlp_` induction sweep, `match_` two-name runs) remove that confound.
- All results are on synthetic data at toy scale; Phase 3 (Pythia-scale corpus subset)
  is untouched.

## Additions (2026-09-13)

- Tier 2 vs tier 1 across all 39 induction runs: Spearman with formation -0.72 (zipper)
  vs -0.93 (reference-model loss); the gap is entirely the vocabulary family. One figure,
  `results/tiers_vs_formation.png`, justifies the tiered framework.
- The induction lag is a free parameter chosen by the data: lag 1 at moderate repeat
  fractions, lags 8-19 with long copies, small vocabularies, or MLPs. The lag sets an exact
  accuracy ceiling (offset > lag). MLP "plateaus" are complete lagged circuits. This is the
  cleanest instance of "the data picks the circuit variant", and it means any formation
  metric tied to the textbook geometry undercounts.
- Selection: zipper-selected 20k subsets form at 700 vs random 5200/never; tier 2 matches
  tier 1 only when its reference model has already formed the circuit.
