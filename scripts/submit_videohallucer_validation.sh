#!/bin/bash
set -euo pipefail
SPLIT=val
DATASET=videohallucer
EVENT_ROOT=${VIDEO_TLDPO_EVENT_ROOT:-/scratch/jp09/dd9648/datasets_video_hallu/eventhallusion}
EVENT_QUESTIONS=${VIDEO_TLDPO_EVENT_QUESTIONS:-}
PREFLIGHT_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --split) SPLIT=${2:?Expected val or test}; shift 2 ;;
        --dataset) DATASET=${2:?Expected dataset}; shift 2 ;;
        --dataset-root) EVENT_ROOT=${2:?Expected dataset path}; shift 2 ;;
        --questions-dir) EVENT_QUESTIONS=${2:?Expected questions path}; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "Usage: bash $0 [--dataset videohallucer|eventhallusion] [--split val|test] [--dataset-root PATH] [--questions-dir PATH] [--preflight-only]" >&2; exit 2 ;;
    esac
done
case "$SPLIT" in val|test) ;; *) echo "Invalid split: $SPLIT" >&2; exit 2 ;; esac
case "$DATASET" in videohallucer|eventhallusion) ;; *) echo "Invalid dataset" >&2; exit 2 ;; esac
export VIDEO_TLDPO_EVAL_SPLIT="$SPLIT"
export VIDEO_TLDPO_EVAL_DATASET="$DATASET"
export VIDEO_TLDPO_EVENT_ROOT="$EVENT_ROOT"
export VIDEO_TLDPO_EVENT_QUESTIONS="$EVENT_QUESTIONS"
JOB_NAME=vtdpo_vh_eval
RESULT_FOLDER=validation_vanilla_dpo_v1
if [ "$SPLIT" = test ]; then
    JOB_NAME=vtdpo_vh_test
    RESULT_FOLDER=test_vanilla_dpo_v1
fi
if [ "$DATASET" = eventhallusion ]; then
    JOB_NAME=vtdpo_eh_eval
    RESULT_FOLDER=eventhallusion_vanilla_dpo_v1
fi
export VIDEO_TLDPO_PROJECT=${VIDEO_TLDPO_PROJECT:-/g/data/hn98/dd9648/Video-TL-DPO}
PROJECT="$VIDEO_TLDPO_PROJECT"
SCRIPT="$PROJECT/pbs/evaluate_videohallucer_validation.pbs"
export VIDEO_TLDPO_ARCHIVE=${VIDEO_TLDPO_ARCHIVE:-/g/data/hn98/envs/dd9648_qwen25_vtdpo_v2.tar.gz}
export VIDEO_TLDPO_MODEL=${VIDEO_TLDPO_MODEL:-/g/data/hn98/dd9648/models/Qwen2.5-VL-7B-Instruct}
ARCHIVE="$VIDEO_TLDPO_ARCHIVE"
bash -n "$SCRIPT"
if [ ! -r "$ARCHIVE" ]; then
    echo "ERROR: missing environment archive: $ARCHIVE" >&2
    exit 1
fi
python3 "$PROJECT/scripts/evaluate_videohallucer_validation.py" --dataset "$DATASET" --split "$SPLIT" --preflight-only
if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    exit 0
fi
ACTIVE=$(qselect -u "$USER" -N "$JOB_NAME")
if [ -n "$ACTIVE" ]; then
    printf 'ERROR: existing evaluation jobs; refusing duplicates:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
export VIDEO_TLDPO_EVAL_VARIANT=base
EXPORTS=VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_VARIANT,VIDEO_TLDPO_EVAL_SPLIT,VIDEO_TLDPO_EVAL_DATASET,VIDEO_TLDPO_EVENT_ROOT,VIDEO_TLDPO_EVENT_QUESTIONS
BASE=$(qsub -r y -N "$JOB_NAME" -v "$EXPORTS" "$SCRIPT")
printf 'base_job=%s\n' "$BASE"
export VIDEO_TLDPO_EVAL_VARIANT=adapter
ADAPTER=$(qsub -r y -N "$JOB_NAME" -W "depend=afterok:$BASE" \
    -v "$EXPORTS" "$SCRIPT")
printf 'adapter_job=%s\n' "$ADAPTER"
echo "result: $PROJECT/outputs/$RESULT_FOLDER/comparison.json"
