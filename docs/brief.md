# Project brief: Mechanism-informed data selection

## Research goal

Study how specific circuits form inside small transformers as a function of
measurable properties of the training data, then use that understanding to
build a data-selection algorithm: given a large corpus, choose a minimal subset
that produces the target circuits and matches full-data performance.

The novel angle: existing data-selection methods (perplexity filtering, DSIR,
DoReMi, RHO-LOSS, dedup) score data against loss or surface statistics. This
project scores data against **mechanism** — which documents cause a specific
circuit to form and perform well.

## Core hypothesis (paper 1)

LZ77 is a hand-coded induction head: it finds the previous occurrence of the
current substring and copies what followed. Therefore the per-document LZ77
compression gain should predict:

1. **When** induction heads form (training step at which activation patching
   recovers ≥50% of clean-vs-corrupted logit difference).
2. **How faithful** the resulting circuit is (accuracy with everything outside
   the circuit ablated, relative to the full model).

Expected: zipper score predicts copying circuits well, fails for compositional
circuits (paper 2).

## Three-tier scoring framework

| Tier | Estimator of C(A\|B) | What it captures | Cost |
|------|---------------------|------------------|------|
| 1. Zipper | (L_{A+B} − L_A) / L_B via LZ77 | Verbatim repeated structure | Trivial |
| 2. LM cross-entropy | Mean loss of B under a small model trained on A | Semantic redundancy (paraphrase, grammar, implication) | Low |
| 3. Circuit-probe learnability | Drop in loss on a circuit-specific probe set after one training step on B | Mechanism contribution | High |

Practical payoff: find out how well tiers 1–2 predict tier 3.

Reference: Baronchelli, Caglioti, Loreto, "Measuring complexity with zippers"
(arXiv physics/0605031) for tier 1; Delétang et al. 2023 "Language Modeling Is
Compression" for the tier 1 ↔ tier 2 equivalence; Mindermann et al. 2022
(RHO-LOSS) for the learnability formulation.

## Experimental plan

### Phase 0 — tooling sanity check
- Reproduce induction heads on GPT-2 small with TransformerLens.
- Verify activation patching and attention visualization work end to end.
- Reference: ARENA curriculum notebooks.

### Phase 1 — data property sweep
- Build a synthetic data generator with independently controllable knobs:
  - **Name-pool entropy** (number/frequency of distinct names) — forces
    generalization vs memorization.
  - **Repetition rate** within context window — known driver of induction.
  - **Conditional entropy of target given context** — task difficulty.
  - **Distractor/filler ratio** — signal dilution.
- Task: two-name "who did what" modeled on Indirect Object Identification.
- Train 2–4 layer transformers, checkpoint every N steps.
- Per checkpoint: activation patching on target behavior → formation step,
  faithfulness, circuit sharpness (heads needed for threshold recovery),
  generalization to unseen names.
- 3 seeds per condition. Compare at matched tokens seen AND matched loss.
- Fix tokenizer/vocab across conditions.
- Compute tier-1 zipper score per dataset; test correlation with formation
  step and faithfulness.

### Phase 2 — minimal corpus for one circuit
- Greedy selector: score candidates by marginal information vs. already-selected
  set A; reject low-entropy boilerplate and irreducible noise; pick highest
  marginal info; append; repeat.
- Baselines: random subset of equal size; perplexity filtering.
- Measure: formation step, faithfulness, held-out task accuracy.

### Phase 3 — transfer
- Does the circuit-minimal corpus match full-data performance on held-out tasks?
- One experiment on a real corpus subset (Pythia-scale) to show the selector
  does something outside synthetic data.

## Predictions to test
- Non-monotonic entropy relationship: too clean → memorization, no circuit;
  moderate → clean early formation; too noisy → late/messy/absent.
- Zipper predicts induction formation; fails for multi-hop circuits.
- Minimal-for-one-circuit data may starve other circuits — track ≥2 circuits
  early to detect interference.
- Aggressive pruning may cause the model to find a different mechanism;
  faithfulness checks across conditions catch this.

## Known pitfalls
- LZ77 entropy estimates converge slowly (~18% error at 8M chars). Use
  sliding-window LZ77 on token IDs, treat scores as rankings not absolutes,
  keep B short relative to A.
- Zipper is blind to paraphrase; LM tier must cover that.
- Reference-model scores drift as the model improves; compare candidates
  against the same checkpoint.
- Keep a held-out task the selector never sees, to avoid overfitting the proxy.

## Engineering conventions
- Python, PyTorch, TransformerLens. Virtual environment.
- Layout: `data/` (generator with entropy as sweepable parameters),
  `train/` (checkpointing), `analysis/` (patching, faithfulness, zipper
  scores), `select/` (greedy selector), `notebooks/` (exploration).
- CPU-first for 2-layer models; Colab GPU for larger runs.
- Every experiment writes a results row: condition, seed, formation step,
  faithfulness, sharpness, generalization, zipper score.

## First milestone
Generate one synthetic dataset and train one 2-layer model end to end on CPU,
with checkpoints saved. Nothing else until that works.

## Before investing months
Literature check: "circuit" + "data selection" / "data attribution" /
"training dynamics" — Anthropic, EleutherAI, Nanda/ARENA orbit — to confirm
nobody has done the mechanism-scored selection.
