#!/bin/bash
# Phase 3 on The Stack: idempotent, resumable. Every stage is skipped if its output exists.
# Usage on the pod (repo cloned to /workspace/LLM-research, volume at /workspace):
#   HF_TOKEN=... bash pod/run_pilot.sh            # data, scoring, pilot (2 arms x 500M tokens)
#   FULL=1 bash pod/run_pilot.sh                 # then: 3 seeds per arm
#   LONG=1 bash pod/run_pilot.sh                 # then: 2 arms x 1B tokens
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-/workspace/data}; POOL=$DATA/pool
LOG=${LOG:-/workspace/results/pod_queue.log}; mkdir -p "$(dirname "$LOG")" "$DATA"
export CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/workspace/checkpoints} RESULTS_DIR=${RESULTS_DIR:-/workspace/results}
# analysis scripts use repo-relative results/ and checkpoints/; point those at the volume
mkdir -p "$CHECKPOINTS_DIR" "$RESULTS_DIR"; ln -sfn "$CHECKPOINTS_DIR" checkpoints; ln -sfn "$RESULTS_DIR" results
STEPS=${STEPS:-30000}        # 30,000 steps x 64 x 256 = 491M tokens
CKPT=${CKPT:-1000}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAIN="--mmap $POOL --steps $STEPS --ckpt-every $CKPT --eval-every $CKPT --batch-size 64 --n-layers 4 --n-heads 8 --d-model 256 --d-head 32 \
  --positional-embedding-type rotary --weight-decay 0.0 --warmup-steps 500 --device cuda --autocast --resume \
  --checkpoints-dir $CHECKPOINTS_DIR --results-dir $RESULTS_DIR"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "stage 1: data"
if [ ! -f ${POOL}_meta.json ]; then
  uv run python pod/stack_tokenize.py --dataset ${DATASET:-bigcode/the-stack-dedup} --data-dir ${DATA_DIR:-data/python} \
    --text-field ${TEXT_FIELD:-content} --out $POOL --n-ctx 256 --target-tokens ${TARGET_TOKENS:-2500000000} 2>&1 | tee -a "$LOG" || true
  [ -f ${POOL}_meta.json ] || { say "stage 1 failed: no ${POOL}_meta.json"; exit 1; }
fi
say "stage 2: scoring"
[ -f ${POOL}_score_report.json ] || uv run python pod/score_pool.py --pool $POOL --frac 0.2 --seeds 0 1 2 --workers ${WORKERS:-8} 2>&1 | tee -a "$LOG"

run_arm() {  # run_arm <arm> <seed> [steps]
  local arm=$1 seed=$2 steps=${3:-$STEPS} run=stack_${1}_s${2}${3:+_$3}
  local idx=${POOL}_arm_${arm}.npy; [ "$arm" = random ] && idx=${POOL}_arm_random_s${seed}.npy
  if [ -f $CHECKPOINTS_DIR/$run/step_${steps}.pt ]; then say "skip training $run"; else
    say "train $run"
    uv run python -m train.train $TRAIN --mmap-index $idx --run $run --seed $seed --steps $steps 2>&1 | tail -3 | tee -a "$LOG"
  fi
  if [ -f $RESULTS_DIR/$run/prefix_matching_summary.json ]; then say "skip scoring $run"; grep -E "formation|final" "$RESULTS_DIR/$run.pm.log" | tee -a "$LOG"; return; fi
  uv run python -m analysis.prefix_matching --run $run --top-k 512 > "$RESULTS_DIR/$run.pm.log" 2>&1 || { tail -5 "$RESULTS_DIR/$run.pm.log" | tee -a "$LOG"; return 1; }
  grep -E "formation|final" "$RESULTS_DIR/$run.pm.log" | tee -a "$LOG"
}

say "stage 3: pilot"
run_arm divgain 0; run_arm random 0
if [ -n "${FULL:-}" ]; then say "stage 4: three seeds per arm"; for s in 1 2; do run_arm divgain $s; run_arm random $s; done; fi
if [ -n "${LONG:-}" ]; then say "stage 5: 1B tokens"; run_arm divgain 0 60000; run_arm random 0 60000; fi
say "DONE"
