#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-8}
METHODS=${METHODS:-"danceopd diffusionopd flowopd"}
export PYTHONPATH="$ROOT/src:$ROOT/opd:$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PYTHON" scripts/prepare_pickapic_prompts.py --quiet

for method in $METHODS; do
  case "$method" in danceopd|diffusionopd|flowopd) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
  save=${SAVE_ROOT:-$ROOT/outputs/opd_pilot}/$method
  mkdir -p "$save/logs"
  echo "[OPD pilot] method=$method warmup=${BENCH_WARMUP:-2} measured=${BENCH_MEASURED:-6}"
  BENCHMARK=1 BENCH_WARMUP=${BENCH_WARMUP:-2} BENCH_MEASURED=${BENCH_MEASURED:-6} \
    SAVE_DIR="$save" LOG_DIR="$save/logs" T_REF=${T_REF:-175.2} \
    "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    "opd/train_${method}_open3.py"
  cat "$save/benchmark_${method}.json"
done
