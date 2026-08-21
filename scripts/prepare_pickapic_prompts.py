#!/usr/bin/env python3
"""Materialize the exact Pick-a-Pic prompt split used by the paper.

The large Pick-a-Pic image dataset is unnecessary for training: DiffusionOPSD
only consumes prompt text.  This script downloads a pinned, text-only Hugging
Face mirror and applies the bundled release recipe.  The recipe keeps the exact
legacy ordering and includes the small number of prompts absent from the mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "data" / "pickapic_recipe.json"
DEFAULT_OUTPUT = ROOT / "data" / "pickapic"
DRAWBENCH_PATH = ROOT / "data" / "drawbench" / "test.txt"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_recipe() -> dict:
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    if recipe.get("format_version") != 1:
        raise RuntimeError(f"unsupported Pick-a-Pic recipe: {recipe.get('format_version')!r}")
    return recipe


def materialize(output: Path, *, force: bool = False) -> tuple[Path, Path]:
    recipe = _load_recipe()
    train_path = output / "train.txt"
    test_path = output / "test.txt"

    expected_hash = str(recipe["paper_train_sha256"])
    if train_path.is_file() and not force:
        actual_hash = _sha256(train_path.read_bytes())
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"existing {train_path} has SHA-256 {actual_hash}, expected {expected_hash}; "
                "rerun with --force to replace it"
            )
        if not test_path.is_file():
            output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DRAWBENCH_PATH, test_path)
        return train_path, test_path

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the project first: pip install -e .") from exc

    dataset = load_dataset(
        recipe["dataset_id"],
        revision=recipe["dataset_revision"],
        split=recipe["dataset_split"],
        # This text mirror is public.  Force anonymous access so an expired
        # login token cannot turn prompt preparation into a misleading 401.
        token=False,
    )
    if len(dataset) != int(recipe["dataset_rows"]):
        raise RuntimeError(f"pinned dataset row count changed: {len(dataset)}")

    column = str(recipe["dataset_column"])
    rows = dataset[column]
    prompts: list[str] = []
    for selector in recipe["selectors"]:
        prompt = rows[selector] if isinstance(selector, int) else selector
        if not isinstance(prompt, str) or "\n" in prompt or "\r" in prompt:
            raise RuntimeError("Pick-a-Pic recipe produced a non-line-safe prompt")
        prompts.append(prompt)

    if len(prompts) != int(recipe["paper_train_lines"]):
        raise RuntimeError(f"prompt count mismatch: {len(prompts)}")
    train_bytes = "".join(f"{prompt}\n" for prompt in prompts).encode("utf-8")
    actual_hash = _sha256(train_bytes)
    if actual_hash != expected_hash:
        raise RuntimeError(f"reconstructed Pick-a-Pic SHA-256 {actual_hash}, expected {expected_hash}")

    output.mkdir(parents=True, exist_ok=True)
    train_path.write_bytes(train_bytes)
    # Trainers expect train.txt/test.txt in one directory.  The paper's held-out
    # evaluation set is DrawBench, so expose that fixed manifest as test.txt.
    shutil.copyfile(DRAWBENCH_PATH, test_path)
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    train_path, test_path = materialize(args.output.resolve(), force=args.force)
    if not args.quiet:
        recipe = _load_recipe()
        print(f"Pick-a-Pic train: {train_path} ({recipe['paper_train_lines']} prompts)")
        print(f"SHA-256: {recipe['paper_train_sha256']}")
        print(f"DrawBench test: {test_path}")


if __name__ == "__main__":
    main()
