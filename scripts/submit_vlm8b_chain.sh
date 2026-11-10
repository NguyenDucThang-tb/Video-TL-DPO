#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND=${1:?Provide qwen25, qwen3 or internvl35}
JOBS=${2:-1}
exec bash "$ROOT/scripts/submit_unified.sh" train "$BACKEND" --jobs "$JOBS"
