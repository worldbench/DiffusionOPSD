#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
METHOD=${METHOD:-${1:-}}
if [[ -z "$METHOD" ]]; then
  echo "Usage: METHOD=<danceopd|diffusionopd|flowopd> bash opd/launch_opd.sh" >&2
  exit 2
fi
case "$METHOD" in danceopd|diffusionopd|flowopd) ;; *) echo "Unknown method: $METHOD" >&2; exit 2 ;; esac

PYTHON=${PYTHON:-python}
NPROC=${NPROC:-8}
SAVE_DIR=${SAVE_DIR:-$ROOT/outputs/opd/$METHOD}
LOG_DIR=${LOG_DIR:-$SAVE_DIR/logs}
export PYTHONPATH="$ROOT/src:$ROOT/opd:$ROOT:${PYTHONPATH:-}"
export SAVE_DIR LOG_DIR

cd "$ROOT"
"$PYTHON" scripts/prepare_pickapic_prompts.py --quiet
echo "[OPD] method=$METHOD policy_ranks=$NPROC save=$SAVE_DIR"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  "opd/train_${METHOD}_open3.py"
