# LLM research: mechanism-informed data selection

How do measurable properties of training data control which circuits form inside small transformers, and can that be turned into a data-selection algorithm? See [CLAUDE.md](CLAUDE.md) for the full research brief, layout and conventions.

## Setup

```bash
uv sync
uv run pytest
```

## First milestone: one dataset, one 2-layer model, on CPU

```bash
uv run python -m data.generator --out datasets/base.npz --name-pool-size 16 --repetition-rate 0.5
uv run python -m train.train --dataset datasets/base.npz --run base_s0 --steps 1500
```

Outputs: `checkpoints/base_s0/step_*.pt`, `results/base_s0/log.jsonl`, and a row in `results/results.csv`.

## Layout

| Path | Purpose |
|---|---|
| `data/` | Synthetic generator with sweepable entropy, repetition, target-noise and distractor knobs, plus held-out-IO-name and held-out-pair generalisation splits |
| `train/` | Training loop with periodic checkpoints and a results row per run |
| `analysis/` | Tier-1 zipper scores, per-checkpoint head-level circuit measurements, sweep summaries. Activation patching and faithfulness not started |
| `selection/` | Greedy mechanism-informed selector (not started) |
| `notebooks/` | Exploration |
| `docs/` | Literature check and notes |
| `datasets/`, `checkpoints/`, `results/` | Local outputs, git-ignored |
