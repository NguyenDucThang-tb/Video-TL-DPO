#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-}"; DATASET="${2:-videohallucer}"; SPLIT="${3:-test}"
case "$BACKEND" in qwen3|internvl35) ;; *) echo "Usage: $0 {qwen3|internvl35} [videohallucer|eventhallusion] [val|test]" >&2; exit 2 ;; esac
case "$DATASET" in videohallucer|eventhallusion) ;; *) echo "Invalid dataset" >&2; exit 2 ;; esac
case "$SPLIT" in val|test) ;; *) echo "Invalid split" >&2; exit 2 ;; esac
PROJECT="${VIDEO_TLDPO_PROJECT:-/g/data/hn98/${USER}/Video-TL-DPO}"
MODEL="${VIDEO_TLDPO_MODEL:-/g/data/hn98/${USER}/models/$([ "$BACKEND" = qwen3 ] && echo Qwen3-VL-8B-Instruct || echo InternVL3_5-8B-Instruct)}"
ARCHIVE="${VIDEO_TLDPO_ARCHIVE:-/g/data/hn98/envs/${USER}_video_vlm8b.tar.gz}"
TRAINOUT="$PROJECT/outputs/vanilla_dpo_${BACKEND}_videohallucer_v1"
EVALROOT="$PROJECT/outputs/${DATASET}_${BACKEND}_vanilla_dpo_v1"
export VIDEO_TLDPO_BACKEND="$BACKEND" VIDEO_TLDPO_MODEL="$MODEL" VIDEO_TLDPO_ARCHIVE="$ARCHIVE"
export VIDEO_TLDPO_EVAL_DATASET="$DATASET" VIDEO_TLDPO_EVAL_SPLIT="$SPLIT"
export VIDEO_TLDPO_TRAINOUT="$TRAINOUT" VIDEO_TLDPO_EVALROOT="$EVALROOT"
export VIDEO_TLDPO_EVAL_VARIANT=base
BASE=$(qsub -r y -v VIDEO_TLDPO_BACKEND,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_DATASET,VIDEO_TLDPO_EVAL_SPLIT,VIDEO_TLDPO_TRAINOUT,VIDEO_TLDPO_EVALROOT,VIDEO_TLDPO_EVAL_VARIANT "$ROOT/pbs/evaluate_videohallucer_backend.pbs")
export VIDEO_TLDPO_EVAL_VARIANT=adapter
ADAPTER=$(qsub -r y -W "depend=afterok:$BASE" -v VIDEO_TLDPO_BACKEND,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_DATASET,VIDEO_TLDPO_EVAL_SPLIT,VIDEO_TLDPO_TRAINOUT,VIDEO_TLDPO_EVALROOT,VIDEO_TLDPO_EVAL_VARIANT "$ROOT/pbs/evaluate_videohallucer_backend.pbs")
echo "base_job=$BASE"
echo "adapter_job=$ADAPTER"
echo "comparison=$EVALROOT/comparison.json"
