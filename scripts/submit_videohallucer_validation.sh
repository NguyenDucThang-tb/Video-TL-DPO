#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=eval; ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-only) MODE=base; shift ;;
    --dataset-root) export VIDEO_TLDPO_EVENT_ROOT=${2:?}; shift 2 ;;
    --questions-dir) export VIDEO_TLDPO_EVENT_QUESTIONS=${2:?}; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec bash "$ROOT/scripts/submit_unified.sh" "$MODE" qwen25 "${ARGS[@]}"
