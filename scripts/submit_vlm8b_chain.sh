#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${VIDEO_TLDPO_PROJECT:-/g/data/hn98/${USER}/Video-TL-DPO}"
BACKEND="${1:-}"
JOBS="${2:-16}"
case "$BACKEND" in qwen3|internvl35) ;; *) echo "Usage: $0 {qwen3|internvl35} [chain_jobs]" >&2; exit 2 ;; esac
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "chain_jobs must be a positive integer" >&2; exit 2; }
MODEL="${VIDEO_TLDPO_MODEL:-/g/data/hn98/${USER}/models/$([ "$BACKEND" = qwen3 ] && echo Qwen3-VL-8B-Instruct || echo InternVL3_5-8B-Instruct)}"
ARCHIVE="${VIDEO_TLDPO_ARCHIVE:-/g/data/hn98/envs/${USER}_video_vlm8b.tar.gz}"
DATA="$PROJECT/artifacts/train_binary.jsonl"

for required in "$DATA" "$PROJECT/artifacts/val_binary.jsonl" \
  "$PROJECT/artifacts/test_records.jsonl" "$PROJECT/artifacts/split_manifest.json" \
  "$MODEL/config.json" "$ARCHIVE"; do
  [[ -r "$required" ]] || { echo "PREFLIGHT FAILED: missing $required" >&2; exit 1; }
done
"${VIDEO_TLDPO_PYTHON:-python3}" "$ROOT/scripts/preflight_vlm8b.py" \
  "$BACKEND" "$MODEL" "$PROJECT/artifacts"
echo "preflight: PASS backend=$BACKEND; records=300; expected val=100,test=310"
PREV=""
IDS="$PROJECT/logs/${BACKEND}_vanilla_chain_$(date +%Y%m%dT%H%M%S)_$$.ids"
mkdir -p "$(dirname "$IDS")"
for ((n=1; n<=JOBS; n++)); do
  export VIDEO_TLDPO_BACKEND="$BACKEND" VIDEO_TLDPO_MODEL="$MODEL"
  export VIDEO_TLDPO_ARCHIVE="$ARCHIVE"
  if [[ -z "$PREV" ]]; then
    JOB=$(qsub -r y -v VIDEO_TLDPO_BACKEND,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE \
      "$ROOT/pbs/train_videohallucer_backend.pbs")
  else
    JOB=$(qsub -r y -W "depend=afterok:$PREV" \
      -v VIDEO_TLDPO_BACKEND,VIDEO_TLDPO_MODEL,VIDEO_TLDPO_ARCHIVE \
      "$ROOT/pbs/train_videohallucer_backend.pbs")
  fi
  printf '%s\n' "$JOB" | tee -a "$IDS"
  PREV="$JOB"
done
echo "job_ids_file=$IDS"
echo "NOTE: qsub acceptance is not training completion; inspect all logs and train_summary.json."
