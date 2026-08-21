import os

# Reward checkpoints (open_clip_pytorch_model.bin, HPS_v2.1_compressed.pt, ...).
# Resolution order:
#   1. REWARD_CKPT_PATH env var
#   2. repo-local ../reward_ckpts (offline / local dev fallback)
CKPT_PATH = os.environ.get(
    "REWARD_CKPT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reward_ckpts"),
)
