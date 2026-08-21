#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/train_public.sh <sd35|zimage> <reward> [extra config flags...]" >&2
  echo "Rewards: hpsv2, clipscore, pickscore, aesthetic, imagereward, hpsv3, deqa; sd35 also supports open3" >&2
  exit 2
fi

BACKBONE=$1
REWARD=$2
shift 2

case "$BACKBONE" in
  sd35) SCRIPT=scripts/train_opsd_ri_sd3.py ;;
  zimage) SCRIPT=scripts/train_opsd_zimage.py ;;
  *) echo "Unknown backbone: $BACKBONE" >&2; exit 2 ;;
esac
case "$REWARD" in
  hpsv2|clipscore|pickscore|aesthetic|imagereward|hpsv3|deqa) ;;
  open3) [[ "$BACKBONE" == "sd35" ]] || { echo "open3 is SD3.5-only" >&2; exit 2; } ;;
  *) echo "Unknown public reward: $REWARD" >&2; exit 2 ;;
esac

HEAVY_ZIMAGE=0
if [[ "$BACKBONE" == "zimage" && ( "$REWARD" == "hpsv3" || "$REWARD" == "deqa" ) ]]; then
  HEAVY_ZIMAGE=1
fi
if [[ -z "${NPROC+x}" ]]; then
  [[ "$HEAVY_ZIMAGE" == 1 ]] && NPROC=7 || NPROC=8
fi
if [[ -z "${UPDATES+x}" ]]; then
  [[ "$BACKBONE" == "sd35" && "$REWARD" == "open3" ]] && UPDATES=300 || UPDATES=100
fi
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${BACKBONE}_${REWARD}}
MODEL=${MODEL:-}
export PUBLIC_N_GPUS="$NPROC"
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
if [[ "$HEAVY_ZIMAGE" == 1 ]]; then
  [[ "$NPROC" == 7 ]] || {
    echo "Paper-matched heavy Z-Image uses NPROC=7 (6 policy ranks + 1 reward server)." >&2
    exit 2
  }
  export ZIMAGE_HEAVY_BRIDGE=0
  export ZIMAGE_HEAVY_DIFF_BRIDGE=1
  export PUBLIC_POLICY_WORLD_SIZE="$((NPROC - 1))"
else
  export ZIMAGE_HEAVY_BRIDGE=0
  export ZIMAGE_HEAVY_DIFF_BRIDGE=0
  export PUBLIC_POLICY_WORLD_SIZE="$NPROC"
fi
export WANDB_MODE=${WANDB_MODE:-offline}

PYTHON=${PYTHON:-python}
"$PYTHON" scripts/prepare_pickapic_prompts.py --quiet
"$PYTHON" scripts/check_reward_setup.py --reward "$REWARD" --backbone "$BACKBONE"

CMD=(
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
  "$SCRIPT"
  --config "config/public.py:${BACKBONE}_${REWARD}"
  --config.num_epochs="$UPDATES"
  --config.save_dir="$OUTPUT_DIR"
)
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  # Preserve the paper's K trajectories per prompt while reducing a pilot to
  # one globally group-complete rollout batch and one optimizer update.
  CMD+=(
    --config.debug=true
    --config.sample.num_batches_per_epoch=1
    --config.train.gradient_accumulation_steps=1
  )
fi
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
CMD+=("$@")

"${CMD[@]}"
