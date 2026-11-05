#!/bin/bash
set -euo pipefail
SPLIT=val
PREFLIGHT_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --split) SPLIT=${2:?Expected val or test}; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "Usage: bash $0 [--split val|test] [--preflight-only]" >&2; exit 2 ;;
    esac
done
case "$SPLIT" in val|test) ;; *) echo "Invalid split: $SPLIT" >&2; exit 2 ;; esac
export VIDEO_TLDPO_EVAL_SPLIT="$SPLIT"
JOB_NAME=vtdpo_vh_eval
RESULT_FOLDER=validation_vanilla_dpo_v1
if [ "$SPLIT" = test ]; then
    JOB_NAME=vtdpo_vh_test
    RESULT_FOLDER=test_vanilla_dpo_v1
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
python3 "$PROJECT/scripts/evaluate_videohallucer_validation.py" --split "$SPLIT" --preflight-only
if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    exit 0
fi
ACTIVE=$(qselect -u "$USER" -N "$JOB_NAME")
if [ -n "$ACTIVE" ]; then
    printf 'ERROR: existing evaluation jobs; refusing duplicates:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
export VIDEO_TLDPO_EVAL_VARIANT=base
BASE=$(qsub -r y -N "$JOB_NAME" -v VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_VARIANT,VIDEO_TLDPO_EVAL_SPLIT "$SCRIPT")
printf 'base_job=%s\n' "$BASE"
export VIDEO_TLDPO_EVAL_VARIANT=adapter
ADAPTER=$(qsub -r y -N "$JOB_NAME" -W "depend=afterok:$BASE" \
    -v VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_VARIANT,VIDEO_TLDPO_EVAL_SPLIT "$SCRIPT")
printf 'adapter_job=%s\n' "$ADAPTER"
echo "result: $PROJECT/outputs/$RESULT_FOLDER/comparison.json"
