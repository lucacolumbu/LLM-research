# LLM research

Mechanistic-interpretability project: how properties of training data shape circuit formation in small transformers, and how to select data by mechanism. Python 3.12+, PyTorch, TransformerLens, managed with `uv`. Full brief below.

## Commands

- `uv sync` installs everything into `.venv`
- `uv run pytest` runs the offline tests (about 30 s, trains a tiny model)
- `uv run ruff check .` lints
- `uv run python -m data.generator --out datasets/<name>.npz [--name-pool-size N --name-zipf S --repetition-rate R --target-noise P --distractor-ratio D --ctx-len L --n-train N --seed K]` generates one condition
- `uv run python -m train.train --dataset datasets/<name>.npz --run <run> [--steps N --ckpt-every N --eval-every N --seed K --d-model D --n-layers L]` trains and checkpoints
- `uv run python -m analysis.zipper --dataset datasets/<name>.npz --update-results` computes the tier-1 zipper score and fills `zipper_score` in results.csv
- `uv run python -m analysis.circuit --run <run>` measures every checkpoint: per-head attention to IO and to S, direct logit attribution per head and per component; writes `results/<run>/circuit.jsonl`, `circuit_summary.json`, `trajectory.png` (`--summary-only` rebuilds the last two)
- `uv run python -m train.sweep --knob name_pool_size --values 4 8 16 32 64 --seeds 0 1 2` runs generate, train, zipper and circuit per (value, seed), 3 jobs in parallel, resumable
- `uv run python -m analysis.summarize --knob name_pool_size --runs "pool*_s*"` aggregates a sweep into `results/summary_<knob>.csv` and `.png`
- `uv run python -m analysis.patching --run <run> --update-results` activation patching on single-sentence prompts across checkpoints: per-head recovery, circuit-head ablation, keep-only-k faithfulness and sharpness, context-bonus decomposition; fills formation_step, faithfulness, sharpness in results.csv
- `uv run python -m data.induction --out datasets/<name>.npz [--repeat-frac F --vocab-size V --noise P]` generates the pure induction task (random tokens with an embedded verbatim repeat); sweep it with `train.sweep --generator data.induction --knob repeat_frac`
- `uv run python -m analysis.leak_curves --runs <fine runs>` plots held-out IO accuracy against leaked examples seen
- `uv run python -m analysis.induction_patching --run <run>` the brief's formation definition on the induction task: source-segment corruption, layer-1 patching at target positions, recovery >= 0.5 with a 1-nat effect floor; writes `induction_patching_summary.json`
- `uv run python -m analysis.summarize ... --correlate zipper_score:formation_step_acc90` adds a Spearman/Pearson correlation across runs
- `uv run python -m analysis.tier2_dataset --reference <ckpt> --runs <runs>` scores datasets under a reference model; `analysis.tiers_figure` plots both tiers against formation
- `uv run python -m selection.select_by_score --pool datasets/pool_het.npz --method zipper|random|refloss|oracle --n N --out datasets/sel_x.npz` selects a subset of a heterogeneous pool by a per-document score
- `uv run python -m selection.experiment --n 20000 --seeds 0 1 2` runs the Phase 2 comparison end to end (pool, arms, training, `results/selection_summary.csv`)
- `train.train --switch-dataset <npz> --switch-step N` trains on a second dataset from step N (data-schedule experiments such as late leaks)
- `uv run jupyter lab` opens notebooks

## Layout

- `data/generator.py`: synthetic IOI-style "who did what" generator. `DataConfig` holds the four sweepable knobs (name-pool entropy via `name_pool_size` + `name_zipf`, `repetition_rate`, `target_noise`, `distractor_ratio`). Fixed word-level vocab shared by all conditions. Generalisation is tested by holding out a *role* or a *combination*, never a token: `n_heldout_io_names` pool names appear in training only as the subject and never as the answer (`heldout_io` split), and `heldout_pair_frac` of the regular pairs never co-occur in training (`heldout_pairs` split).
- `train/train.py`: trains a `HookedTransformer` (default 2 layers, d_model 128, 4 heads) with AdamW on CPU. Saves `checkpoints/<run>/step_<N>.pt` every `ckpt_every` steps, logs eval metrics to `results/<run>/log.jsonl`, and appends one row to `results/results.csv`. `load_checkpoint()` rebuilds a model from a checkpoint.
- `analysis/zipper.py`: tier 1. `Zipper.conditional_many` gives C(B|A) reusing one zlib compressor state for A; `compression_gain` is per-document gain against a shuffled copy (unigram skew does not count). Tiers 2 and 3 not started.
- `analysis/circuit.py`: head-level measurements at IO positions across checkpoints. Exact direct-logit-attribution decomposition (heads, MLPs, embeddings, attention biases, and the LN-bias plus unembed-bias "prior" term); `decomposed_ld` must equal `logit_diff`. Formation steps are defined on the final top-attribution heads. Activation patching, faithfulness and sharpness not started.
- `analysis/lm_scores.py`: tier 2, mean cross-entropy per document under a checkpoint; `analysis/tier2_dataset.py` turns it into a per-dataset gain. Tier 3 (circuit-probe learnability) not started.
- `analysis/plots.py`, `analysis/summarize.py`, `analysis/zipper_figure.py`, `analysis/leak_curves.py`: static matplotlib figures; fixed categorical palette, one measure per axis.
- `selection/select_by_score.py`, `selection/experiment.py`: Phase 2 minimal selector and its driver. The greedy marginal-information selector from the brief is not started.
- `data/induction.py`: pure induction task. Same npz layout; only train and val splits. `s_ids` is the current token at each target, so `attn_s` = duplicate-token attention and `attn_io` = attention to the token to be copied.
- `analysis/patching.py`: activation patching and ablations on 17-token single-sentence IOI prompts (answer read at index 14). Mean ablation is per position (prompts share one template); a single mean over positions is off-distribution and inflates sharpness. Use `--k-circuit 0` (smallest head set recovering 80% by joint patching) and `--greedy-max 12` (greedy forward selection for sufficiency); keep-only-k by single-head ranking is kept only for reference.
- `train/sweep.py`: one-knob sweep runner (`--generator` selects the task module); run names are `<short knob><value>_s<seed>` (e.g. `pool16_s0`, `leak0.045_s0`).
- `selection/`: greedy selector. Named `selection` because a top-level `select/` package would shadow the stdlib `select` module.
- `notebooks/`: exploration only.
- `datasets/`, `checkpoints/`, `results/` are git-ignored outputs.

## Conventions

- Results row columns are fixed in `train.train.RESULTS_COLUMNS`: condition, seed, steps, tokens seen, losses, IO accuracy and logit diff, generalization (IO accuracy on the `heldout_io` split, the headline), generalization_pairs (`heldout_pairs` split), then formation_step, faithfulness, sharpness, zipper_score. Training leaves the last four empty for `analysis/` to fill.
- Dataset files carry their full `DataConfig`, vocab and stats in `meta`; never rely on filenames to recover a condition.
- IO metrics are computed at target positions: `target_mask` marks the IO token, `s_ids` gives the subject at that position, and the prediction comes from the logits one position earlier.
- Training and generation are seeded; the results row records the seed. Use 3 seeds per condition.
- Matched architecture for both tasks: 2 layers, 4 heads, MLPs, rotary positions, weight decay 0, warmup 500. With weight decay 0.01 (or standard learned positions) the induction circuit does not form in an MLP model within budget; attention-only models form it under either setting. Original sweeps used attention-only rotary for induction and standard positions with weight decay 0.01 for the two-name task; the `mlp_` and `match_` runs are the matched replications.
- Trace one trained model's mechanism with edge knockouts before sweeping any new synthetic task. Three shortcuts were found this way (see docs/findings.md).
- The induction lag is data-dependent. A lag-k circuit (layer-0 head carrying the token k+1 back, layer-1 head reading k after the needed token) is a complete copy mechanism with an accuracy ceiling equal to the fraction of targets whose copy offset exceeds k. Lag 1 appears at moderate repeat fractions in attention-only models; lags 8-19 appear with long copies, small vocabularies, and in every MLP model. `attn_io` (lag 0 only) misses lagged circuits; use `formation_step_attn_lag` / `dominant_lag` from `analysis.circuit` (needs `need_pos`, reconstructed by `data.induction.need_positions`) or the behavioural `formation_step_acc50`.
- Tier-2 dataset scores (`analysis/tier2_dataset.py`, loss gain vs shuffled documents under a fixed reference model with an induction head) repair the zipper's blind spot on the vocabulary axis; both tiers agree on every other axis. Reference: `checkpoints/sel_random_s0/step_6000.pt`.
- Two tasks, two mechanisms, by design. The two-name task is solved by S-inhibition heads plus MLP-mediated promotion of in-context names (no attention copying); the induction task forces attention copying. Test the LZ77 hypothesis on the induction task; use the two-name task for interference studies.
- Do not assume the circuit is a name-mover. In the base condition the top-attribution heads attend to the subject and suppress it (S-inhibition), no head attends to the IO, and the held-out-IO failure is carried by MLP 0, not the unembed bias. Record attention to both IO and S.
- Never test generalisation on tokens absent from training: their output weights are untrained and the model cannot emit them, so the score is zero regardless of mechanism.
- `HookedTransformer` is marked deprecated in TransformerLens 3.x in favour of `TransformerBridge`, but the bridge only wraps pretrained HuggingFace models. Keep using `HookedTransformer` for from-scratch training; the deprecation warning is expected.
- CPU first. Keep default runs under a few minutes; move bigger sweeps to Colab.

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

-
