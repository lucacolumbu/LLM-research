# Can we pick training data so that a circuit forms early?

A research project on how circuits form in small transformers and how to select
training data for them. The headline result: a per-document score computed with
`gzip` predicts when induction heads form, and selecting data by it makes them
form two to seven times sooner than a random subset of the same size, on
synthetic data and on 2.5B tokens of Python from The Stack.

- **Two-page summary:** [docs/summary_2page.pdf](docs/summary_2page.pdf)
- **Paper draft:** [docs/main_v04.pdf](docs/main_v04.pdf) (source in `docs/main.tex`)
- **Lab notebook:** [docs/findings.md](docs/findings.md), every experiment with its numbers, dated, including the ones that failed
- **Original brief:** [docs/brief.md](docs/brief.md)

## The idea in one paragraph

An induction head looks back to the previous occurrence of the current token and
predicts what followed it. The LZ77 algorithm inside `gzip` does the same thing
to compress a file. So a document's *compression gain*, how much shorter it
compresses than a shuffled copy of itself, measures how much it exercises an
induction head, for free and without a model. The project tests whether that
score predicts when the circuit forms, whether it can select a smaller training
set that forms the circuit sooner, and what such a selection costs.

## Main results

| Finding | Evidence |
|---|---|
| Compression gain predicts the step at which induction heads form | Spearman −0.98 (attention-only), −0.92 (with MLPs), −1.0 on condition means, three seeds, replicated at 3L8H |
| Selecting a third of a mixed pool by gain forms the circuit 7x sooner than random | Step 700 in every seed vs 5,200 or never; matches a copy-length oracle (97% subset overlap) |
| The data decides which *variant* of the circuit forms | A family of lag-k induction circuits with exact accuracy ceilings, traced to a positional copier the model builds first |
| A reference model repairs the compressor's blind spot | Vocabulary family: ρ from −0.72 to −0.93 pooled over 39 runs |
| Compression distance (diversity) is a separate quantity | Decorrelated from gain; relates to generalisation across held-out pairs, not to formation |
| On The Stack, gain + diversity selection forms induction heads 2–3x sooner than random | 1,000 / 2,000 steps in every seed vs 1,500–2,000 / 4,000–6,000; 0.2 nats of validation loss as the cost, stable at 1B tokens |

![Gain predicts formation](figures/zipper_vs_formation.png)

![The Stack: selected vs random](figures/stack_trajectories.png)

More figures in [figures/](figures/).

## Reproduce one experiment

```bash
uv sync
uv run pytest                                   # offline tests, about 30 s

# one induction dataset, one 2-layer model, the circuit measured at every checkpoint
uv run python -m data.induction --out datasets/ind.npz --repeat-frac 0.75
uv run python -m train.train --dataset datasets/ind.npz --run ind_s0 --steps 3000
uv run python -m analysis.zipper --dataset datasets/ind.npz --update-results
uv run python -m analysis.circuit --run ind_s0

# the repeat-fraction sweep behind the headline correlation (three seeds, three parallel jobs)
uv run python -m train.sweep --generator data.induction --knob repeat_frac --values 0.35 0.5 0.75 0.98 --seeds 0 1 2
uv run python -m analysis.summarize --knob repeat_frac --runs "rep*_s*" --correlate zipper_score:formation_step_acc50

# the selection experiment: pool, four arms, training, summary
uv run python -m selection.experiment --n 20000 --seeds 0 1 2
```

Everything synthetic runs on a laptop CPU. The Stack experiment ran on one rented
RTX 4090 for about $7; its pipeline is in `pod/` and its plan in
`docs/runpod_plan.md`.

## Layout

| Path | Purpose |
|---|---|
| `data/` | Two synthetic tasks: the pure induction task (`induction.py`) and the two-name IOI-style task (`generator.py`), each with independently controllable data knobs |
| `train/` | Training with checkpoints every N steps, a results row per run, memmapped pools for real corpora, resumable runs; `sweep.py` runs one knob across seeds |
| `analysis/` | Compression scores (`zipper.py`), per-checkpoint circuit measurements (`circuit.py`), activation patching and sufficient sets (`patching.py`, `induction_patching.py`), prefix-matching, lag tracing, in-context delta and swap controls, figures |
| `selection/` | Selection by score, the diversity-constrained greedy selector, and the end-to-end selection experiment |
| `pod/` | Tokenising, scoring and training The Stack on a Runpod GPU; results retrieved through the log stream |
| `tests/` | Offline tests for the generators, trainer, scores, circuit analysis, patching and selection |
| `docs/` | Paper, summary, findings log, brief, literature notes, Runpod plan |
| `figures/` | Final figures |
| `results/stack/` | The Stack runs' trajectories and analyses, the only results tracked |

`CLAUDE.md` holds the working conventions used while the project was built with
Claude Code as a pair; it is kept as part of the record.
