#!/usr/bin/env bash
set -euo pipefail

TARGET=${REWARD_CKPT_PATH:-"$(pwd)/reward_ckpts"}
mkdir -p "$TARGET"
export REWARD_CKPT_PATH="$TARGET"

command -v hf >/dev/null 2>&1 || {
  echo "Missing 'hf' CLI. Install/upgrade huggingface-hub first." >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || {
  echo "Missing curl." >&2
  exit 2
}

echo "Downloading public reward assets to $TARGET"
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  open_clip_pytorch_model.bin --local-dir "$TARGET"
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir "$TARGET"
curl --fail --location --retry 3 \
  'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac%2Blogos%2Bava1-l14-linearMSE.pth' \
  --output "$TARGET/sac+logos+ava1-l14-linearMSE.pth"

echo "Reward assets ready. Export:"
echo "  export REWARD_CKPT_PATH='$TARGET'"
