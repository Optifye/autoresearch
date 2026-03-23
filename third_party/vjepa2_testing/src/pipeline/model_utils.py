"""Model helpers for frozen encoder + attentive classifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from evals.video_classification_frozen.models import init_module
from vjepa2.src.models.attentive_pooler import AttentiveClassifier

from src import constants


def _resolve_checkpoint(config) -> str:
    """Resolve checkpoint with optional env/explicit override."""
    env_override = os.getenv("VJEPA_ENCODER_CHECKPOINT") or os.getenv("VJEPA_ENCODER_CKPT")
    if env_override:
        return str(Path(env_override).expanduser())
    return str(Path(config["checkpoint"]).expanduser())


def load_frozen_encoder(
    device: torch.device | None = None,
    logger: logging.Logger | None = None,
    checkpoint: str | None = None,
):
    log = logger or logging.getLogger(__name__)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = constants.ENCODER_CONFIG
    variant = constants.SELECTED_ENCODER
    enc_cfg = (config.get("model_kwargs", {}) or {}).get("encoder", {}) or {}
    requested_model = enc_cfg.get("model_name") or variant.model_name
    if requested_model not in {variant.model_name} | {"vit_giant", "vit_giant_xformers", "vit_large"}:
        raise ValueError(
            f"Unsupported encoder model_name '{requested_model}'. "
            f"Allowed: {variant.model_name} (selected via VJEPA_ENCODER_MODEL={variant.key})."
        )
    enc_cfg["model_name"] = variant.model_name
    enc_cfg["checkpoint_key"] = variant.checkpoint_key
    config["model_kwargs"] = {"encoder": enc_cfg}
    ckpt_path = str(Path(checkpoint).expanduser()) if checkpoint else _resolve_checkpoint(config)
    ckpt_path_resolved = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path_resolved.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint missing at {ckpt_path_resolved} for variant {variant.model_name}. "
            "Set VJEPA_ENCODER_CHECKPOINT to the correct path."
        )
    log.info(
        "Loading encoder checkpoint %s (module=%s, model=%s) on device %s",
        ckpt_path,
        config["module_name"],
        enc_cfg.get("model_name"),
        target_device,
    )
    encoder = init_module(
        module_name=config["module_name"],
        frames_per_clip=config["frames_per_clip"],
        resolution=config["resolution"],
        checkpoint=str(ckpt_path_resolved),
        model_kwargs=config["model_kwargs"],
        wrapper_kwargs=config["wrapper_kwargs"],
        device=target_device,
    )
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    embed_dim = getattr(encoder, "embed_dim", None)
    num_heads = getattr(encoder, "num_heads", None)
    depth = None
    base_model = getattr(encoder, "model", None)
    if base_model is not None and hasattr(base_model, "blocks"):
        depth = len(getattr(base_model, "blocks", []))
    if embed_dim != variant.embed_dim or num_heads != variant.num_heads or (depth not in {None, variant.depth}):
        raise ValueError(
            "Encoder architecture mismatch after load: "
            f"got embed_dim={embed_dim}, heads={num_heads}, depth={depth}; "
            f"expected embed_dim={variant.embed_dim}, heads={variant.num_heads}, depth={variant.depth} "
            f"for variant {variant.model_name}."
        )
    log.info(
        "Encoder loaded and validated: variant=%s model=%s embed_dim=%s heads=%s depth=%s resolution=%s",
        variant.key,
        variant.model_name,
        embed_dim,
        num_heads,
        depth,
        config.get("resolution"),
    )
    return encoder


def build_classifier(embed_dim: int, device: torch.device) -> AttentiveClassifier:
    cfg = constants.CLASSIFIER_CONFIG
    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_classes=constants.NUM_CLASSES,
        num_heads=cfg.get("num_heads", 16),
        depth=cfg.get("num_probe_blocks", 4),
        use_activation_checkpointing=True,
    ).to(device)
    if cfg.get("freeze_pooler", False):
        for param in classifier.pooler.parameters():
            param.requires_grad = False
        logging.getLogger(__name__).info("Classifier pooler frozen; training linear head only.")
    return classifier


def _unwrap_classifier_state_dict(payload) -> dict:
    """Unwrap common checkpoint formats into a flat state_dict.

    Supported formats:
    - Vendor checkpoints: {"classifier": state_dict, ...}
    - Packaged poolers: {"pooler_state": state_dict, "metadata": {...}}
    - SSV2 pretrained poolers: {"classifiers": [state_dict]} (often with DDP "module." prefixes)
    """

    if isinstance(payload, dict):
        candidate = payload.get("classifier")
        if isinstance(candidate, dict):
            return candidate
        candidate = payload.get("pooler_state")
        if isinstance(candidate, dict):
            return candidate
        classifiers = payload.get("classifiers")
        if isinstance(classifiers, list) and classifiers:
            first = classifiers[0]
            if isinstance(first, dict):
                return first
        # Fallback: some callers may pass a raw state_dict directly.
        return payload
    raise TypeError(f"Unsupported pooler checkpoint payload type: {type(payload)}")


def load_pooler_weights(classifier: AttentiveClassifier, payload, logger: logging.Logger) -> None:
    state_dict = _unwrap_classifier_state_dict(payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict mapping, got {type(state_dict)}")

    # Normalize common checkpoint prefixes (DDP).
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items() if isinstance(k, str)}

    stripped = [k for k in state_dict.keys() if isinstance(k, str) and k.startswith("linear.")]
    pooler_state = {k: v for k, v in state_dict.items() if isinstance(k, str) and not k.startswith("linear.")}
    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    if stripped:
        logger.info("Skipped %d linear tensors when loading pretrained pooler", len(stripped))
    missing_non_linear = [k for k in missing if isinstance(k, str) and not k.startswith("linear.")]
    unexpected_non_linear = [k for k in unexpected if isinstance(k, str) and not k.startswith("linear.")]
    logger.info(
        "Pooler load summary: provided=%d pooler_keys=%d missing_non_linear=%d unexpected_non_linear=%d",
        len(state_dict),
        len(pooler_state),
        len(missing_non_linear),
        len(unexpected_non_linear),
    )
    if missing_non_linear:
        logger.warning("Pooler load missing non-linear keys (sample): %s", missing_non_linear[:5])
    if unexpected_non_linear:
        logger.warning("Pooler load unexpected non-linear keys (sample): %s", unexpected_non_linear[:5])
