#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MIXED_REWARDS='reward[=weight],...' bash scripts/train_mixed_reward.sh <opsd|nft> [extra config flags...]" >&2
  exit 2
fi

METHOD=$1
shift
case "$METHOD" in
  opsd) SCRIPT=scripts/train_opsd_ri_sd3.py ;;
  nft) SCRIPT=scripts/train_nft_sd3.py ;;
  *) echo "Unknown mixed-reward method: $METHOD (choose opsd or nft)" >&2; exit 2 ;;
esac

# Paper example: PickScore/26 + CLIPScore + HPSv2.1. The PickScore scorer
# applies /26 internally, so all three objective weights are one.
export PUBLIC_MIXED_REWARDS=${MIXED_REWARDS:-pickscore=1,clipscore=1,hpsv2=1}
NPROC=${NPROC:-8}
UPDATES=${UPDATES:-300}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/mixed_${METHOD}}
MODEL=${MODEL:-}
PYTHON=${PYTHON:-python}

[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC must be a positive integer" >&2; exit 2; }
[[ "$UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "UPDATES must be a positive integer" >&2; exit 2; }

export PUBLIC_N_GPUS="$NPROC"
export PUBLIC_POLICY_WORLD_SIZE="$NPROC"
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0
export ZIMAGE_HEAVY_DIFF_BRIDGE=0
export WANDB_MODE=${WANDB_MODE:-offline}

"$PYTHON" scripts/prepare_pickapic_prompts.py --quiet

# Let the config parser validate names, duplicates, and weights, then check
# every selected reward's dependencies before torchrun starts.
REWARD_NAMES=$(
  "$PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

path = Path("config/mixed.py")
spec = importlib.util.spec_from_file_location("_mixed_config_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("\n".join(module.parse_reward_spec()))
PY
)
while IFS= read -r reward; do
  [[ -n "$reward" ]] || continue
  "$PYTHON" scripts/check_reward_setup.py --reward "$reward" --backbone sd35
done <<< "$REWARD_NAMES"

CMD=(
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
  "$SCRIPT"
  --config "config/mixed.py:sd35_${METHOD}"
  --config.num_epochs="$UPDATES"
  --config.save_dir="$OUTPUT_DIR"
)
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  CMD+=(
    --config.debug=true
    --config.num_epochs=1
    --config.sample.num_batches_per_epoch=1
    --config.train.gradient_accumulation_steps=1
  )
fi
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
CMD+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
