"""Master entrypoint for downloading data and training the classifier."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from pathlib import Path


# Ensure the original V-JEPA repo modules are importable
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT / "vjepa2"

if REPO_ROOT.exists():
    repo_pkg = str(REPO_ROOT)
    if repo_pkg not in sys.path:
        sys.path.insert(0, repo_pkg)

if WORKSPACE_ROOT.exists():
    root_str = str(WORKSPACE_ROOT)
    if root_str not in sys.path:
        sys.path.append(root_str)

import torch
from torch.utils.data import DataLoader

import constants
import inference as inference_utils
from pipeline.clip_validator import validate_clips
from pipeline.datasets import create_datasets
from pipeline.logging_utils import setup_logger
from pipeline.model_utils import build_classifier, load_frozen_encoder, load_pooler_weights
from pipeline.s3_utils import S3Downloader
from pipeline.trainer import ClassifierTrainer


def ensure_directories(logger: logging.Logger):
    for path in [constants.RAW_CLIP_DIR, constants.VALIDATED_CLIP_DIR, constants.CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Output directories -> raw:%s validated:%s checkpoints:%s",
        constants.RAW_CLIP_DIR,
        constants.VALIDATED_CLIP_DIR,
        constants.CHECKPOINT_DIR,
    )


def download_clips(logger: logging.Logger):
    downloader = S3Downloader(logger)
    counts = downloader.download_many(constants.S3_CLASS_PATHS, constants.RAW_CLIP_DIR)
    logger.info("Download counts per class: %s", counts)
    return counts


def validate_data(logger: logging.Logger):
    class_dirs = {name: constants.RAW_CLIP_DIR / name for name in constants.CLASS_NAMES}
    validated = validate_clips(class_dirs, constants.NUM_FRAMES_IN_CLIP, logger)
    summary = {k: len(v) for k, v in validated.items()}
    logger.info("Validated clip counts: %s", summary)
    return validated


def build_dataloaders(validated, logger: logging.Logger, explicit_split=None):
    train_ds, val_ds, labels = create_datasets(
        validated,
        train_split=constants.TRAIN_SPLIT,
        seed=constants.RANDOM_SEED,
        logger=logger,
        explicit_split=explicit_split,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Insufficient data after validation. Check downloaded clips.")

    train_loader = DataLoader(
        train_ds,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=constants.NUM_WORKERS,
        pin_memory=constants.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=constants.NUM_WORKERS,
        pin_memory=constants.PIN_MEMORY,
    )

    logger.info(
        "DataLoaders ready -> train batches: %d, val batches: %d",
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader, labels


def _load_explicit_split(logger: logging.Logger):
    split_path = os.getenv("VJEPA_FOLD_SPLIT_PATH")
    if not split_path:
        return None
    candidate = Path(split_path)
    if not candidate.exists():
        logger.warning("VJEPA_FOLD_SPLIT_PATH set to %s but file not found; ignoring.", split_path)
        return None
    try:
        raw = json.loads(candidate.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse fold split file %s: %s", split_path, exc)
        return None
    logger.info("Loaded explicit fold split from %s", candidate)
    return raw


def _maybe_warm_start_pooler(
    *,
    classifier,
    device: torch.device,
    logger: logging.Logger,
    freeze_pooler: bool,
    pooler_ckpt: Path,
) -> None:
    # Only warm-start the shared/trainable pooler; head-only runs already load the provided pooler checkpoint.
    if freeze_pooler:
        return
    if pooler_ckpt.exists():
        logger.info("Skipping pretrained pooler init because checkpoint already exists at %s", pooler_ckpt)
        return
    use_flag = os.getenv("VJEPA_USE_PRETRAINED_POOLER", "1").strip().lower()
    if use_flag not in {"1", "true", "yes", "on"}:
        logger.info("Pretrained pooler init disabled via VJEPA_USE_PRETRAINED_POOLER=%s", use_flag)
        return
    init_path_env = os.getenv("VJEPA_INIT_POOLER_CHECKPOINT")
    if not init_path_env:
        logger.info("VJEPA_USE_PRETRAINED_POOLER enabled but VJEPA_INIT_POOLER_CHECKPOINT not set; using random init")
        return
    init_path = Path(init_path_env)
    if not init_path.exists():
        logger.warning(
            "VJEPA_INIT_POOLER_CHECKPOINT=%s but file not found; proceeding with random init",
            init_path,
        )
        return
    try:
        state_raw = inference_utils._load_checkpoint_state(init_path, device, logger)  # pylint: disable=protected-access
        pooler_state = inference_utils._extract_component_state(  # pylint: disable=protected-access
            state_raw,
            "pooler",
            classifier.pooler.state_dict().keys(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load pretrained pooler from %s: %s", init_path, exc)
        return
    if not pooler_state:
        logger.warning("No pooler weights found in %s; using random init", init_path)
        return
    load_result = classifier.pooler.load_state_dict(pooler_state, strict=False)
    inference_utils._log_load_warnings(logger, "Pretrained pooler", load_result)  # pylint: disable=protected-access
    logger.info("Initialized attentive pooler from pretrained checkpoint %s", init_path)


def run_pipeline():
    logger = setup_logger()
    enc = constants.SELECTED_ENCODER
    logger.info(
        "Encoder variant resolved: key=%s model=%s ckpt=%s key=%s expected_dim=%s heads=%s depth=%s resolution=%s frames=%s",
        enc.key,
        enc.model_name,
        enc.checkpoint,
        enc.checkpoint_key,
        enc.embed_dim,
        enc.num_heads,
        enc.depth,
        constants.ENCODER_CONFIG.get("resolution"),
        constants.NUM_FRAMES_IN_CLIP,
    )
    logger.info("Configured clip length: %d frames", constants.NUM_FRAMES_IN_CLIP)
    logger.info("=== Stage 1/5: Preparing workspace ===")
    ensure_directories(logger)

    logger.info("=== Stage 2/5: Downloading raw clips ===")
    download_clips(logger)

    logger.info("=== Stage 3/5: Validating clip integrity ===")
    validated = validate_data(logger)
    explicit_split = _load_explicit_split(logger)
    if explicit_split:
        remapped = {}
        for class_name, buckets in explicit_split.items():
            train_names = set(buckets.get("train", []))
            val_names = set(buckets.get("val", []))
            available = {p.name: p for p in validated.get(class_name, [])}
            remapped[class_name] = {
                "train": [str(available[name]) for name in train_names if name in available],
                "val": [str(available[name]) for name in val_names if name in available],
            }
        explicit_split = remapped
        logger.info("Using explicit fold split for datasets (classes=%d)", len(explicit_split))

    logger.info("=== Stage 4/5: Building datasets and loaders ===")
    train_loader, val_loader, labels = build_dataloaders(validated, logger, explicit_split=explicit_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cuda_raw = os.getenv("VJEPA_REQUIRE_CUDA", "0")
    require_cuda = require_cuda_raw.strip().lower() in {"1", "true", "yes", "on"}
    if require_cuda and device.type != "cuda":
        msg = "VJEPA_REQUIRE_CUDA=1 but CUDA is unavailable"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Using device: %s", device)

    encoder = load_frozen_encoder(device, logger)
    classifier = build_classifier(encoder.embed_dim, device)

    classifier_cfg = constants.CLASSIFIER_CONFIG
    freeze_pooler = classifier_cfg.get("freeze_pooler", False)
    pooler_ckpt = Path(classifier_cfg.get("pooler_checkpoint", constants.POOLER_CHECKPOINT_PATH))
    head_ckpt = Path(classifier_cfg.get("head_checkpoint", constants.LINEAR_HEAD_CHECKPOINT_PATH))
    target_ckpt = head_ckpt if freeze_pooler else pooler_ckpt

    _maybe_warm_start_pooler(
        classifier=classifier,
        device=device,
        logger=logger,
        freeze_pooler=freeze_pooler,
        pooler_ckpt=pooler_ckpt,
    )

    if freeze_pooler:
        if pooler_ckpt.exists():
            payload = torch.load(pooler_ckpt, map_location=device)
            load_pooler_weights(classifier, payload, logger)
            logger.info("Loaded pretrained pooler weights from %s for head-only fine-tuning", pooler_ckpt)
        else:
            logger.warning(
                "freeze_pooler=True but pooler checkpoint %s not found; proceeding without pretrained weights",
                pooler_ckpt,
            )
        logger.info("Fine-tuned checkpoints will be saved to %s", target_ckpt)
    else:
        logger.info("Pooler is trainable; checkpoints will be saved to %s", target_ckpt)

    logger.info(
        "=== Stage 5/5: Training classifier (%d classes) ===",
        len(labels),
    )
    trainer = ClassifierTrainer(
        encoder=encoder,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        checkpoint_path=target_ckpt,
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train attentive classifier on custom clips")
    parser.add_argument("--run", action="store_true", help="No-op flag to mirror CLI style")
    _ = parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
