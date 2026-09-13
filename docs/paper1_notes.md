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
- Architecture matters for which circuit the data can buy: with MLPs the induction
  circuit did not form within budget on this data (learning-rate check pending).

## Caveats to state

- The two-name task is at 2 layers and 4 heads; sparsity numbers there are not
  meaningful. A 4-layer 8-head replication is queued.
- The two tasks use different architectures (MLPs vs attention-only); cross-task claims
  are about mechanism, not about matched models.
- All results are on synthetic data at toy scale; Phase 3 (Pythia-scale corpus subset)
  is untouched.
