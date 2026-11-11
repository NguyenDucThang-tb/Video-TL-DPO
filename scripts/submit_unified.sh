#!/usr/bin/env bash
# No model/env/data downloads and no GPU work on the login node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIDEO_TLDPO_PROJECT="${VIDEO_TLDPO_PROJECT:-/g/data/hn98/$USER/Video-TL-DPO}"
MODE=${1:-}; BACKEND=${2:-}; shift "$(( $# >= 2 ? 2 : $# ))"
case "$MODE" in base|train|eval) ;; *) echo "Usage: $0 {base|train|eval} {qwen25|qwen3|internvl35} [--jobs N] [--dataset NAME] [--split val|test] [--preflight-only]" >&2; exit 2 ;; esac
case "$BACKEND" in qwen25|qwen3|internvl35) ;; *) echo "Invalid backend" >&2; exit 2 ;; esac
JOBS=1; DATASET=both; SPLIT=test; PREFLIGHT=0; WALLTIME=00:30:00
while [ "$#" -gt 0 ]; do
  case "$1" in
    --jobs) JOBS=${2:?}; shift 2 ;;
    --dataset) DATASET=${2:?}; shift 2 ;;
    --split) SPLIT=${2:?}; shift 2 ;;
    --preflight-only) PREFLIGHT=1; shift ;;
    --walltime) WALLTIME=${2:?}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] && [ "$JOBS" -le 32 ] || { echo "jobs must be 1..32" >&2; exit 2; }
if [ "$MODE" != train ] && [ "$JOBS" != 1 ]; then
  echo "--jobs chains apply to training; partial eval must be explicitly resubmitted" >&2; exit 2
fi
case "$DATASET" in both) DATASETS=(videohallucer eventhallusion) ;; videohallucer|eventhallusion) DATASETS=("$DATASET") ;; *) echo "Invalid dataset" >&2; exit 2 ;; esac
case "$SPLIT" in val|test) ;; *) echo "Invalid split" >&2; exit 2 ;; esac
[[ "$WALLTIME" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || { echo "walltime must be HH:MM:SS" >&2; exit 2; }
export VIDEO_TLDPO_JOB_SECONDS=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
[ "$VIDEO_TLDPO_JOB_SECONDS" -gt 300 ] || { echo "walltime must exceed five minutes" >&2; exit 2; }
if [ "$MODE" != train ] && [ "$SPLIT" = val ] && [ "$DATASET" != videohallucer ]; then
  echo "VH_VAL calibration uses --dataset videohallucer --split val" >&2; exit 2
fi
export VIDEO_TLDPO_BACKEND="$BACKEND" VIDEO_TLDPO_EVAL_SPLIT="$SPLIT"
COMMAND=evaluate; VARIANT=base
if [ "$MODE" = train ]; then COMMAND=train; DATASETS=(videohallucer); fi
if [ "$MODE" = eval ]; then VARIANT=adapter; fi
export VIDEO_TLDPO_COMMAND="$COMMAND" VIDEO_TLDPO_EVAL_VARIANT="$VARIANT"
PYTHON=${VIDEO_TLDPO_PYTHON:-python3}
for DATA in "${DATASETS[@]}"; do
  ARGS=("$COMMAND" --project "$VIDEO_TLDPO_PROJECT" --backend "$BACKEND" --dataset "$DATA" --split "$SPLIT" --variant "$VARIANT" --preflight-only)
  if [ -n "${VIDEO_TLDPO_MODEL:-}" ]; then ARGS+=(--model "$VIDEO_TLDPO_MODEL"); fi
  if [ -n "${VIDEO_TLDPO_EVENT_QUESTIONS:-}" ]; then ARGS+=(--questions-dir "$VIDEO_TLDPO_EVENT_QUESTIONS"); fi
  if [ -n "${VIDEO_TLDPO_OUT:-}" ]; then ARGS+=(--output "$VIDEO_TLDPO_OUT"); fi
  "$PYTHON" "$ROOT/scripts/run_unified.py" "${ARGS[@]}"
done
if [ "$PREFLIGHT" = 1 ]; then echo "preflight_only: no jobs submitted"; exit 0; fi
case "$BACKEND" in qwen25) DEFAULT_ARCHIVE=/g/data/hn98/envs/${USER}_qwen25_vtdpo_v2.tar.gz ;; *) DEFAULT_ARCHIVE=/g/data/hn98/envs/${USER}_video_vlm8b.tar.gz ;; esac
export VIDEO_TLDPO_ARCHIVE=${VIDEO_TLDPO_ARCHIVE:-$DEFAULT_ARCHIVE}
test -r "$VIDEO_TLDPO_ARCHIVE" || { echo "Missing env archive" >&2; exit 1; }
for DATA in "${DATASETS[@]}"; do
  ACTIVE=$(qselect -u "$USER" -N "vtdpo3_${BACKEND}_${MODE}_${DATA}_${SPLIT}")
  if [ -n "$ACTIVE" ]; then
    echo "Existing jobs for $BACKEND/$MODE/$DATA; refusing duplicates: $ACTIVE" >&2
    exit 1
  fi
done
EXPORTS=VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_BACKEND,VIDEO_TLDPO_COMMAND,VIDEO_TLDPO_EVAL_VARIANT,VIDEO_TLDPO_EVAL_DATASET,VIDEO_TLDPO_EVAL_SPLIT,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_JOB_SECONDS
for NAME in VIDEO_TLDPO_MODEL VIDEO_TLDPO_CANONICAL_SPLIT VIDEO_TLDPO_CANONICAL_EVAL VIDEO_TLDPO_CANONICAL_TRAIN VIDEO_TLDPO_EVENT_ROOT VIDEO_TLDPO_EVENT_QUESTIONS VIDEO_TLDPO_OUT VIDEO_TLDPO_RUN_ROOT; do
  if [ -n "${!NAME:-}" ]; then export "$NAME"; EXPORTS+=",$NAME"; fi
done
LOGDIR="$VIDEO_TLDPO_PROJECT/logs"
mkdir -p "$LOGDIR"
IDS="$LOGDIR/unified_${BACKEND}_${MODE}_$(date +%Y%m%dT%H%M%S)_$$.ids"
for DATA in "${DATASETS[@]}"; do
  export VIDEO_TLDPO_EVAL_DATASET="$DATA"
  PREV=""
  for ((N=1; N<=JOBS; N++)); do
    DEPENDENCY=()
    if [ -n "$PREV" ]; then DEPENDENCY=(-W "depend=afterok:$PREV"); fi
    JOB=$(qsub -r y -l "walltime=$WALLTIME" -N "vtdpo3_${BACKEND}_${MODE}_${DATA}_${SPLIT}" "${DEPENDENCY[@]}" -v "$EXPORTS" "$ROOT/pbs/run_unified.pbs")
    printf '%s\n' "$JOB" | tee -a "$IDS"
    PREV="$JOB"
  done
done
echo "job_ids_file=$IDS"
echo "Results: ${VIDEO_TLDPO_RUN_ROOT:-$VIDEO_TLDPO_PROJECT/outputs/unified_adaptive_seed_v3}/$BACKEND/"
