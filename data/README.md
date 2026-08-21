# Prompt data

- `pickapic_recipe.json`: release recipe for the exact 25,415-prompt Pick-a-Pic training manifest used by the paper. It pins the text-only [`sayakpaul/pick-a-pic-v2-unique-prompts`](https://huggingface.co/datasets/sayakpaul/pick-a-pic-v2-unique-prompts) revision, preserves the legacy ordering, and records the expected SHA-256.
- `drawbench/test.txt`: the 1,000-prompt DrawBench evaluation manifest used by the paper's fixed held-out protocol.

Materialize the training manifest (about 3 MB; no Pick-a-Pic images are downloaded):

```bash
python scripts/prepare_pickapic_prompts.py
```

This creates ignored files `data/pickapic/{train,test}.txt`. The public training launchers run the command automatically. To use another one-prompt-per-line dataset, pass `--config.dataset=/path/to/dataset` directly to the trainer.
