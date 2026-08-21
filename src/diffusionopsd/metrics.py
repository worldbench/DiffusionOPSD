"""Shared scalar-metric instrumentation for training entry points."""

from __future__ import annotations

import json
import os
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
import time
from typing import Any, Mapping


def install_wandb_jsonl_tee(
    wandb_module: Any,
    metrics_path: str | Path,
    *,
    durable: bool = False,
    strict: bool = False,
):
    """Mirror JSON-compatible scalar ``wandb.log`` payloads to JSONL.

    The public trainers all emit the same ``metrics.jsonl`` schema.  Keeping the
    wrapper here prevents method-specific copies from drifting while leaving image,
    histogram, and other non-scalar WandB values untouched.
    """

    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_log = wandb_module.log
    write_lock = Lock()

    def write_record(record: Mapping[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with write_lock:
            if not durable:
                with path.open("ab") as stream:
                    stream.write(line)
                return

            # ReFL resumes from metrics.jsonl, so publish its record through an
            # atomic replace and make retries idempotent. Other trainers retain
            # the cheaper append-only path above.
            last_error = None
            for attempt in range(5):
                tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
                try:
                    existing = path.read_bytes() if path.exists() else b""
                    if existing and not existing.endswith(b"\n"):
                        raise RuntimeError(f"refusing to append to truncated JSONL: {path}")
                    if existing and existing.rsplit(b"\n", 2)[-2] + b"\n" == line:
                        return
                    with tmp.open("wb") as stream:
                        stream.write(existing + line)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp, path)
                    return
                except OSError as exc:
                    last_error = exc
                    tmp.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(0.25 * (2**attempt))
            if last_error is not None:  # pragma: no cover - loop always returns/raises
                raise last_error

    def log_with_jsonl(data: Mapping[str, Any], step=None, **kwargs):
        try:
            record = {}
            for key, value in data.items():
                if isinstance(value, bool):
                    record[key] = value
                elif isinstance(value, Integral):
                    record[key] = int(value)
                elif isinstance(value, Real):
                    record[key] = float(value)
                elif isinstance(value, (str, type(None))):
                    record[key] = value
            if record:
                record["_step"] = step
                write_record(record)
        except Exception:
            if strict:
                raise
            # Normal trainer metrics must never interrupt a model update. ReFL
            # opts into strict durable writes because resume correctness uses it.
        return original_log(data, step=step, **kwargs)

    wandb_module.log = log_with_jsonl
    return log_with_jsonl
