#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/train_refl.sh <sd35|zimage> <reward> [extra config flags...]" >&2
  echo "Rewards: hpsv2, clipscore, pickscore, aesthetic, imagereward, hpsv3, deqa" >&2
  exit 2
fi

BACKBONE=$1
REWARD=$2
shift 2
case "$BACKBONE" in
  sd35) CONFIG_BACKBONE=sd35m; SCRIPT=scripts/train_refl_sd35m.py ;;
  zimage) CONFIG_BACKBONE=zimage; SCRIPT=scripts/train_refl_zimage.py ;;
  *) echo "Unknown backbone: $BACKBONE" >&2; exit 2 ;;
esac
case "$REWARD" in
  hpsv2|clipscore|pickscore|aesthetic|imagereward|hpsv3|deqa) ;;
  *) echo "Unknown public reward: $REWARD" >&2; exit 2 ;;
esac

HEAVY_ZIMAGE=0
if [[ "$BACKBONE" == "zimage" && ( "$REWARD" == "hpsv3" || "$REWARD" == "deqa" ) ]]; then
  HEAVY_ZIMAGE=1
fi
if [[ -z "${NPROC+x}" ]]; then
  [[ "$HEAVY_ZIMAGE" == 1 ]] && NPROC=7 || NPROC=8
fi
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
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
export WANDB_MODE=${WANDB_MODE:-offline}
PYTHON=${PYTHON:-python}
UPDATES=${UPDATES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/refl/${CONFIG_BACKBONE}_${REWARD}}
MODEL=${MODEL:-}

"$PYTHON" scripts/prepare_pickapic_prompts.py --quiet
"$PYTHON" scripts/check_reward_setup.py --reward "$REWARD" --backbone "$BACKBONE"

CMD=(
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
  "$SCRIPT"
  --config "config/refl.py:${CONFIG_BACKBONE}_${REWARD}"
  --config.refl.num_updates="$UPDATES"
  --config.num_epochs="$UPDATES"
  --config.save_dir="$OUTPUT_DIR"
)
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
CMD+=("$@")
"${CMD[@]}"
