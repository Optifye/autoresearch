"""Minimal attentive-pooler loader for token-mode train.py experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .vjepa.runtime import prioritize_vendor_paths

LOGGER = logging.getLogger(__name__)


def _load_pooler_state_for_classifier(pooler_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(pooler_path, map_location="cpu")
    if isinstance(payload, dict):
        if isinstance(payload.get("pooler_state"), dict):
            state = payload["pooler_state"]
        elif isinstance(payload.get("classifier"), dict):
            state = payload["classifier"]
        elif isinstance(payload.get("classifiers"), list) and payload.get("classifiers"):
            first = payload["classifiers"][0]
            state = first if isinstance(first, dict) else None
        else:
            state = payload if payload else None
    else:
        state = None
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported pooler checkpoint format: {pooler_path}")

    if any(isinstance(key, str) and key.startswith("module.") for key in state.keys()):
        state = {key[len("module.") :]: value for key, value in state.items() if isinstance(key, str)}
    if not any(isinstance(key, str) and key.startswith("pooler.") for key in state.keys()):
        state = {
            (
                key
                if (isinstance(key, str) and (key.startswith("pooler.") or key.startswith("linear.")))
                else f"pooler.{key}"
            ): value
            for key, value in state.items()
        }
    state = {key: value for key, value in state.items() if isinstance(key, str) and not key.startswith("linear.")}
    if not state:
        raise RuntimeError(f"No pooler tensors found in {pooler_path}")
    return state


def build_probe_pooler(input_embed_dim: int, device: torch.device, pooler_path: Path) -> nn.Module:
    prioritize_vendor_paths()
    from third_party.vjepa2_testing.src.pipeline.model_utils import build_classifier

    classifier = build_classifier(int(input_embed_dim), device)
    pooler_state = _load_pooler_state_for_classifier(pooler_path)
    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    if missing or unexpected:
        LOGGER.info("Probe load warnings missing=%s unexpected=%s", missing[:8], unexpected[:8])
    return classifier.pooler.to(device)
