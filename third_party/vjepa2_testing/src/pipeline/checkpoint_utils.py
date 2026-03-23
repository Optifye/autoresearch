"""Checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_classifier_checkpoint(
    classifier_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    epoch: int,
    best_metric: float,
    output_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    if output_path is None and output_dir is None:
        raise ValueError("Either output_path or output_dir must be provided.")
    if output_path is None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "latest.pt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "classifier": classifier_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "metric": best_metric,
    }
    torch.save(payload, output_path)
    metadata = {"epoch": epoch, "metric": best_metric}
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    return output_path
