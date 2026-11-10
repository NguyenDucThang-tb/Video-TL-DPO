#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND=${1:?Provide backend}; DATASET=${2:-videohallucer}; SPLIT=${3:-test}
MODE=eval
if [ "${4:-}" = --base-only ]; then MODE=base; fi
exec bash "$ROOT/scripts/submit_unified.sh" "$MODE" "$BACKEND" --dataset "$DATASET" --split "$SPLIT"
