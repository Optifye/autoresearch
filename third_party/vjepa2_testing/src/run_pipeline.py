"""End-to-end runner that trains the classifier and launches inference from one JSON config."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping

import constants
import inference as inference_module
import main as training_main
from pipeline.aws_credentials import get_s3_client
from pipeline.s3_utils import parse_s3_uri


def _configure_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("pipeline-runner")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_class_paths(raw: Mapping[str, Any]) -> Dict[str, List[str]]:
    if not raw:
        raise ValueError("training.class_s3_paths must contain at least one class")
    normalized: Dict[str, List[str]] = {}
    for label, value in raw.items():
        if isinstance(value, str):
            paths = [value]
        elif isinstance(value, (list, tuple)):
            paths = [str(item) for item in value if str(item).strip()]
        else:
            raise TypeError(f"Unsupported type for class '{label}': {type(value)!r}")
        if not paths:
            raise ValueError(f"Class '{label}' must include at least one S3 path")
        normalized[label] = paths
    return normalized


def _apply_training_config(cfg: Dict[str, Any], run_name: str | None, logger: logging.Logger) -> Dict[str, Any]:
    class_paths = _normalize_class_paths(cfg.get("class_s3_paths", {}))
    constants.S3_CLASS_PATHS = class_paths
    constants.CLASS_NAMES = list(class_paths.keys())
    constants.NUM_CLASSES = len(constants.CLASS_NAMES)

    checkpoint_subdir = cfg.get("checkpoint_subdir") or run_name or constants.DEFAULT_CLASSIFIER_SUBDIR
    checkpoint_dir = constants.PROJECT_ROOT / "artifacts" / "classifiers" / checkpoint_subdir

    data_run_root = constants.DATA_ROOT / checkpoint_subdir
    raw_dir = data_run_root / "raw"
    validated_dir = data_run_root / "validated"
    constants.RAW_CLIP_DIR = raw_dir
    constants.VALIDATED_CLIP_DIR = validated_dir

    constants.CHECKPOINT_DIR = checkpoint_dir
    constants.POOLER_CHECKPOINT_PATH = checkpoint_dir / "pooler_pretrained.pt"
    constants.LINEAR_HEAD_CHECKPOINT_PATH = checkpoint_dir / "linear_head_latest.pt"
    constants.CHECKPOINT_PATH = constants.LINEAR_HEAD_CHECKPOINT_PATH
    constants.LOG_FILENAME = checkpoint_dir / "training.log"
    constants.CLASSIFIER_CONFIG["pooler_checkpoint"] = str(constants.POOLER_CHECKPOINT_PATH)
    constants.CLASSIFIER_CONFIG["head_checkpoint"] = str(constants.LINEAR_HEAD_CHECKPOINT_PATH)

    freeze_pooler = bool(cfg.get("freeze_pooler", constants.CLASSIFIER_CONFIG.get("freeze_pooler", False)))
    constants.CLASSIFIER_CONFIG["freeze_pooler"] = freeze_pooler
    target_ckpt = constants.LINEAR_HEAD_CHECKPOINT_PATH if freeze_pooler else constants.POOLER_CHECKPOINT_PATH
    constants.CHECKPOINT_PATH = target_ckpt

    enable_movement_augs = bool(
        cfg.get(
            "enable_movement_augmentations",
            constants.AUGMENTATION_CONFIG.get("enable_movement_transforms", True),
        )
    )
    constants.AUGMENTATION_CONFIG["enable_movement_transforms"] = enable_movement_augs

    logger.info(
        "Training config -> %d classes | freeze_pooler=%s | movement_augs=%s",
        constants.NUM_CLASSES,
        freeze_pooler,
        enable_movement_augs,
    )
    logger.info("Checkpoints -> %s | data raw=%s | data validated=%s", checkpoint_dir, raw_dir, validated_dir)
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": target_ckpt,
        "freeze_pooler": freeze_pooler,
        "run_folder": checkpoint_subdir,
    }


def _configure_inference(cfg: Dict[str, Any], checkpoint_path: Path, run_folder: str, logger: logging.Logger) -> Dict[str, Any]:
    roi = cfg.get("roi")
    if not roi or len(roi) != 4:
        raise ValueError("inference.roi must contain four numeric values [x, y, w, h]")
    roi_box = tuple(float(value) for value in roi)
    if cfg.get("results_s3_path") is None:
        raise ValueError("inference.results_s3_path is required")

    inference_module.ROI_BOX = roi_box  # type: ignore[attr-defined]
    inference_module.RESULTS_S3_URI = str(cfg["results_s3_path"])  # type: ignore[attr-defined]
    inference_module.OUTPUT_VIDEO_DIR = constants.PROJECT_ROOT / "artifacts" / "inference" / run_folder  # type: ignore[attr-defined]
    inference_module.CHECKPOINT_PATH = checkpoint_path  # type: ignore[attr-defined]

    logger.info(
        "Inference config -> roi=%s | source=%s",
        roi_box,
        cfg.get("video_s3_path"),
    )
    return {
        "video_s3_path": cfg.get("video_s3_path"),
        "start_time": float(cfg.get("start_time", 0.0)),
        "duration": float(cfg.get("duration", 0.0)),
        "checkpoint_path": checkpoint_path,
    }


def _upload_file_to_s3(source: Path, destination: str, logger: logging.Logger):
    bucket, key = parse_s3_uri(destination)
    logger.info("Uploading %s -> s3://%s/%s", source, bucket, key)
    get_s3_client().upload_file(str(source), bucket, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run inference from a JSON config")
    parser.add_argument("--config", type=Path, required=True, help="Path to the pipeline JSON config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity for the runner (defaults to INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = _configure_logger(args.log_level)
    config_path = args.config.resolve()
    config = _load_config(config_path)

    run_name = config.get("run_name")
    training_cfg = config.get("training") or {}
    inference_cfg = config.get("inference") or {}
    uploads_cfg = config.get("uploads") or {}

    logger.info("Loaded config '%s'", config_path)
    training_state = _apply_training_config(training_cfg, run_name, logger)

    logger.info("=== Stage 1/3: Training classifier ===")
    training_main.run_pipeline()
    logger.info("Training finished. Best checkpoint -> %s", training_state["checkpoint_path"])

    logger.info("=== Stage 2/3: Running inference ===")
    inference_state = _configure_inference(
        inference_cfg,
        Path(inference_cfg.get("checkpoint_path", training_state["checkpoint_path"])),
        training_state["run_folder"],
        logger,
    )
    if not inference_state["video_s3_path"]:
        raise ValueError("inference.video_s3_path is required")
    if inference_state["duration"] <= 0:
        raise ValueError("inference.duration must be greater than zero")
    if inference_state["start_time"] < 0:
        raise ValueError("inference.start_time must be >= 0")

    checkpoint_path = Path(inference_state["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inference_module.run_inference(
        s3_path=inference_state["video_s3_path"],
        start_time=inference_state["start_time"],
        duration=inference_state["duration"],
        checkpoint_path=checkpoint_path,
    )

    dest = uploads_cfg.get("config_s3_path")
    if dest:
        logger.info("=== Stage 3/3: Uploading run config ===")
        _upload_file_to_s3(config_path, dest, logger)
    else:
        logger.info("No config upload path provided; skipping upload")

    logger.info("Pipeline run complete")


if __name__ == "__main__":
    main()
