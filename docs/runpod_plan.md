# Runpod plan: Phase 3 on The Stack at 500M-1B tokens

Status: plan only. No pod, volume or download has been created. Account state at planning
time: no pods, no network volumes.

## Design (as agreed)
Same design as the local code Phase 3 with fresh data at a budget where induction can form.
Pool ~2.5B tokens of Python; score gain, NCD and the match-length histogram; pilot two arms
(gain + diversity constraint vs random, one seed, 4-layer model, 500M tokens, checkpoints
on a fixed cadence, prefix-matching and validation loss at each); decide; then three seeds
per arm and optionally 1B tokens for the winner; analyse as before plus the identifier delta;
copy final checkpoints and results back.

## GPU: A40 Secure at $0.35/h, fallback RTX 4090 Community at $0.34/h
Live catalogue (CUDA 12.8): A40 48 GB, $0.35/h, stock Low, Secure Cloud only; RTX 4090
24 GB, $0.34/h, stock Low; L40S $0.79/h Low; A100 80GB $1.19/h Low; L4 out of stock.
The A40 costs the same as the 4090, has twice the VRAM, and sits in Secure Cloud, which is
not reclaimable, so the resume path becomes insurance rather than a requirement. For a
4-layer d_model-256 model either card is kernel-launch bound; the difference is minutes.
Both show Low stock: if neither is available at launch, L40S at $0.79/h is the next step
and still keeps the whole exercise under $10.

## Budget and time (revised from the local step rate)
The local runs used 8k tokens per step because the 2-layer model was CPU-bound at batch 64,
context 128. On the pod the model is 4L8H d256, context 256, batch 128 = 32k tokens per
step; 500M tokens is 15,000 steps. A 4-layer d256 model does a step of that size in about
40-60 ms on an A40 or 4090, so a 500M-token run is 10-15 minutes and a 1B run 20-30. The
GPU is not the cost; the CPU-side data work is.

| stage | wall time | pod cost at $0.35/h |
|---|---|---|
| download 10 GB of Python source (streaming, stop at 2.5B tokens) | 30-60 min | $0.30 |
| tokenise to a uint16 memmap (8 workers, regex tokeniser) | 5-10 min | |
| gain per document over 20M documents (multiprocessing) | 10-15 min | |
| diversity-constrained greedy over the top fifth (chunked, 16 probes) | 15-30 min | |
| match-length histogram and NCD diversity on samples | 5 min | |
| pilot: 2 arms x 500M tokens + prefix-matching/loss at 30 checkpoints | 40 min | $0.25 |
| decision, then 6 arms x 500M | 1.5-2 h | $0.70 |
| optional: 2 arms x 1B | 1 h | $0.35 |
| identifier delta and swap control on final checkpoints | 20 min | |
| total | 5-7 h of pod time | $2-3 |

Network volume 40 GB at $0.07/GB/month: $2.80/month; delete when done. Egress free.
Terminate the pod between sessions.

## Data
- Primary: `bigcode/the-stack-dedup`, `data/python` parquet shards, streamed with the
  `datasets` library. Gated: needs a Hugging Face token whose account has accepted the
  dataset terms, passed to the pod as `HF_TOKEN`. Stop after 2.5B tokens (about 60 shards).
- Fallback if the token is not ready: `codeparrot/github-code-clean` filtered to Python,
  not gated, same pipeline.
- Documents: files chunked into non-overlapping 256-token windows (context 256, longer than
  the local 128 so more repeats fall inside the window). Validation: 20k held-out windows
  from held-out files; also a repeated-random-sequence probe set for prefix matching.

## Tokeniser decision
Word-level regex tokeniser as locally (identifiers, numbers, string words, one token per
punctuation character, newline token), vocabulary 32,768 by frequency from the first 200M
tokens. Keeps identifiers as single tokens so the prefix-matching probe, the in-context
delta and the swap control transfer unchanged, and pushes UNK below about 1.5% (7% at 8k
locally). A BPE variant is a later comparison, not part of this run.

## Selection at scale
- Gain: zlib compression gain against a shuffled copy, per document, multiprocessing.
- Diversity constraint: greedy by gain in chunks of 10k documents; within a chunk the 16
  probes are fixed (drawn from the accepted set before the chunk), so the chunk is
  parallel; a document is accepted if its NCD to every probe is >= 0.75. Same rule as
  locally, chunking only removes the strictly sequential dependency.
- Arms for the pilot: constrained and random, 20% of the pool each (about 4M documents,
  1B tokens, so 500M tokens is half an epoch and 1B one epoch; no repetition of the
  subset, unlike the 5-15 epochs locally).
- Report before training: gain distribution, match-length histogram of pool vs selected
  (the top fifth locally had 45% of tokens in matches >= 8; confirm the same range), mean
  pairwise NCD of each arm.

## Model and training
4 layers, 8 heads, d_model 256, d_head 32, MLPs, rotary, weight decay 0, warmup 500,
lr 1e-3, batch 128, context 256, bf16 autocast, uint16 memmap sampling. Checkpoints every
500 steps (16M tokens; 30 per 500M run, about 40 MB each) with optimizer state, so a
reclaimed or restarted pod resumes from the last one. Prefix-matching and validation loss
computed at every checkpoint on the pod (GPU, seconds each).

## Decision rule after the pilot
Pull-away = the constrained arm's best-head prefix-matching score exceeds random's by
>= 0.1 at the final checkpoint and at each of the last five, or either arm crosses 0.5.
Then: three seeds per arm; if one arm crossed 0.5, push both to 1B to measure formation
step and the identifier delta after formation. Flat or converging: stop, report the budget
statement with the trajectories.

## What to copy back
Final checkpoint of every run, the checkpoint at first prefix-matching >= 0.5 if any,
`log.jsonl`, prefix-matching summaries, the scoring report and the selection indices
(so arms are reproducible). Leave the memmap and intermediate checkpoints on the volume
until the volume is deleted.

## Pipeline work before the pod (local, no cost)
1. `train/train.py`: memmap dataset path (`--mmap datasets/x.bin --n-ctx 256`, uint16,
   random window sampling), optimizer state in checkpoints, `--resume` from the latest
   checkpoint, bf16 autocast on CUDA, periodic prefix-matching hook.
2. `pod/stack_tokenize.py`: stream the dataset, build the vocabulary, write
   `pool.bin` (uint16), `pool_index.npy` (window offsets), `val.bin`, `vocab.json`.
3. `pod/score_pool.py`: gain per document, chunked constrained greedy, random arms,
   histograms and NCD report, arm index files.
4. `pod/run_pilot.sh`: idempotent end to end (skips finished stages), `pod/run_full.sh`
   for the six-arm follow-up, both resumable.
5. Local smoke test of the memmap path and resume on the existing code pool.

## Pod recipe (to execute only after confirmation)
- Network volume 40 GB in a data centre that has the A40 (or 4090) in stock.
- Pod: `runpod/pytorch` CUDA 12.8 template, 1x A40 Secure, volume mounted at /workspace,
  env `HF_TOKEN`; on boot: clone the repo, `uv sync`, `zsh pod/run_pilot.sh`.
- Progress: `results/pod_queue.log` on the volume, tailed through the pod log stream.
- Terminate the pod at the end of each session; the volume keeps everything.
