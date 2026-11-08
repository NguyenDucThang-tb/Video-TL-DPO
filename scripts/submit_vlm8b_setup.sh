#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qsub "$ROOT/pbs/setup_vlm8b_assets.pbs"
