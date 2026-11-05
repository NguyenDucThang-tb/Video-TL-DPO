#!/bin/bash
set -euo pipefail
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
python3 "$PROJECT/scripts/evaluate_videohallucer_validation.py" --preflight-only
if [ "${1:-}" = "--preflight-only" ]; then
    exit 0
elif [ "$#" -ne 0 ]; then
    echo "Usage: bash $0 [--preflight-only]" >&2
    exit 2
fi
ACTIVE=$(qselect -u "$USER" -N vtdpo_vh_eval)
if [ -n "$ACTIVE" ]; then
    printf 'ERROR: existing evaluation jobs; refusing duplicates:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
export VIDEO_TLDPO_EVAL_VARIANT=base
BASE=$(qsub -r y -v VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_VARIANT "$SCRIPT")
printf 'base_job=%s\n' "$BASE"
export VIDEO_TLDPO_EVAL_VARIANT=adapter
ADAPTER=$(qsub -r y -W "depend=afterok:$BASE" \
    -v VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_EVAL_VARIANT "$SCRIPT")
printf 'adapter_job=%s\n' "$ADAPTER"
echo "result: $PROJECT/outputs/validation_vanilla_dpo_v1/comparison.json"
