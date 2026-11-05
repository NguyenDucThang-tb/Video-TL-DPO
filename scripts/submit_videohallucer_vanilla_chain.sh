#!/bin/bash
# Run on a Gadi login node. Preflight uses only Python's standard library.
set -euo pipefail

JOBS=16
PREFLIGHT_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --jobs) JOBS=${2:?Provide --jobs COUNT}; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "Usage: bash $0 [--jobs COUNT] [--preflight-only]" >&2; exit 2 ;;
    esac
done
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ] || [ "$JOBS" -gt 32 ]; then
    echo "ERROR: --jobs must be between 1 and 32" >&2
    exit 2
fi

export VIDEO_TLDPO_PROJECT=${VIDEO_TLDPO_PROJECT:-/g/data/hn98/dd9648/Video-TL-DPO}
export VIDEO_TLDPO_DATA=${VIDEO_TLDPO_DATA:-$VIDEO_TLDPO_PROJECT/artifacts/train_binary.jsonl}
export VIDEO_TLDPO_MODEL=${VIDEO_TLDPO_MODEL:-/g/data/hn98/dd9648/models/Qwen2.5-VL-7B-Instruct}
export VIDEO_TLDPO_ARCHIVE=${VIDEO_TLDPO_ARCHIVE:-/g/data/hn98/envs/dd9648_qwen25_vtdpo_v2.tar.gz}
export VIDEO_TLDPO_OUT=${VIDEO_TLDPO_OUT:-$VIDEO_TLDPO_PROJECT/outputs/vanilla_dpo_qwen25_vh_continuous_v2}
SCRIPT="$VIDEO_TLDPO_PROJECT/pbs/train_videohallucer_vanilla_resumable.pbs"
for INPUT in "$SCRIPT" "$VIDEO_TLDPO_ARCHIVE" \
    "$VIDEO_TLDPO_PROJECT/scripts/train_videohallucer_vanilla_resumable.py"; do
    if [ ! -r "$INPUT" ]; then
        echo "ERROR: missing/unreadable input: $INPUT" >&2
        exit 1
    fi
done
bash -n "$SCRIPT"
if ! grep -q '^SCHEMA = 2$' "$VIDEO_TLDPO_PROJECT/scripts/train_videohallucer_vanilla_resumable.py" \
    || ! grep -q -- '--deadline-epoch' "$SCRIPT"; then
    echo "ERROR: Python/PBS files are stale; copy both updated files before submitting" >&2
    exit 1
fi

python3 - "$VIDEO_TLDPO_DATA" "$VIDEO_TLDPO_MODEL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, model = map(Path, sys.argv[1:])
try:
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows) != 300 or len({r['sample_id'] for r in rows}) != 300:
        raise ValueError('Expected 300 records with unique sample_id')
    missing = []
    for r in rows:
        if r.get('split') != 'train' or not r.get('question'):
            raise ValueError('Invalid split/question: ' + r['sample_id'])
        chosen = r['chosen'].strip().rstrip('.').lower()
        rejected = r['rejected'].strip().rstrip('.').lower()
        if {chosen, rejected} != {'yes', 'no'} or chosen != r['gt_answer'].strip().lower():
            raise ValueError('Invalid binary preference: ' + r['sample_id'])
        if not Path(r['video_path']).is_file():
            missing.append(r['video_path'])
    if missing:
        raise ValueError('Missing videos: ' + repr(missing[:5]))
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_vl':
        raise ValueError('Model is not Qwen2.5-VL')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards:
        raise ValueError('Empty model weight index')
    required = list(shards) + ['preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json']
    for name in required:
        if not (model / name).is_file():
            raise ValueError('Missing model file: ' + str(model / name))
except (OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit('PREFLIGHT FAILED: ' + str(exc))
print('preflight: PASS; records=300; video_paths_exist=300; model_shards=' + str(len(shards)))
print('dataset_sha256: ' + hashlib.sha256(data.read_bytes()).hexdigest())
PY

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    echo "preflight_only: no jobs submitted"
    exit 0
fi
command -v qsub >/dev/null
command -v qselect >/dev/null
ACTIVE=$(qselect -u "$USER" -N vtdpo_vh_resume)
if [ -n "$ACTIVE" ]; then
    printf 'ERROR: existing vtdpo_vh_resume jobs; refusing a duplicate chain:\n%s\n' "$ACTIVE" >&2
    exit 1
fi

mkdir -p "$VIDEO_TLDPO_PROJECT/logs"
LEDGER="$VIDEO_TLDPO_PROJECT/logs/vanilla_chain_$(date +%Y%m%dT%H%M%S)_$$.ids"
PREV=""
for ((N=1; N<=JOBS; N++)); do
    OPTIONS=(-r y -v VIDEO_TLDPO_PROJECT,VIDEO_TLDPO_DATA,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE,VIDEO_TLDPO_OUT)
    if [ -n "$PREV" ]; then
        OPTIONS+=(-W "depend=afterok:$PREV")
    fi
    if ! JOB=$(qsub "${OPTIONS[@]}" "$SCRIPT"); then
        echo "ERROR: submission stopped at job $N; earlier IDs are in $LEDGER" >&2
        exit 1
    fi
    if ! [[ "$JOB" =~ ^[0-9]+\.[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: unexpected qsub output: $JOB; inspect qstat before retrying" >&2
        exit 1
    fi
    printf '%s\n' "$JOB" >> "$LEDGER"
    printf 'chain_job_%s=%s\n' "$N" "$JOB"
    PREV="$JOB"
done
echo "job_ids_file: $LEDGER"
