"""Inference script for ROI-based attentive classifier."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT / "vjepa2"

if REPO_ROOT.exists():
    repo_pkg = str(REPO_ROOT)
    if repo_pkg not in sys.path:
        sys.path.insert(0, repo_pkg)

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

try:  # pragma: no cover - import guard for optional dependency
    import cv2
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError("OpenCV (cv2) is required for annotation") from exc

from evals.video_classification_frozen.utils import make_transforms

from src import constants
from src.pipeline.aws_credentials import get_s3_client
from src.pipeline.logging_utils import setup_logger
from src.pipeline.model_utils import build_classifier, load_frozen_encoder
from src.pipeline.s3_utils import parse_s3_uri


# ----------------------------- Configuration ---------------------------------
ROI_BOX = (0.2, 0.0, 0.6, 0.85)  # (x, y, w, h) normalized
CLIP_FRAMES = constants.NUM_FRAMES_IN_CLIP
WINDOW_STRIDE = max(1, CLIP_FRAMES // 2)
BATCH_SIZE = 16
CHECKPOINT_PATH = constants.CHECKPOINT_PATH
OUTPUT_VIDEO_DIR = constants.PROJECT_ROOT / "artifacts" / "inference"
RESULTS_S3_URI = "s3://optiframe-video-clips/results/vjepa"
ANNOTATED_TEMPLATE = "{stem}_start{start:.2f}_dur{duration:.2f}_annotated.mp4"
METADATA_TEMPLATE = "{stem}_start{start:.2f}_dur{duration:.2f}_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier inference on an S3 video segment")
    parser.add_argument("--s3-path", required=True, help="Full S3 URI of the source video (e.g., s3://bucket/key.mp4)")
    parser.add_argument("--start-time", type=float, required=True, help="Segment start time in seconds")
    parser.add_argument("--duration", type=float, required=True, help="Segment duration in seconds")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=f"Override classifier checkpoint path (defaults to {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--pooler-checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint containing only the attentive pooler weights. Must be used with "
        "--linear-head-checkpoint-path. When provided, overrides --checkpoint-path.",
    )
    parser.add_argument(
        "--linear-head-checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint containing only the linear head weights. Must be used with "
        "--pooler-checkpoint-path. When provided, overrides --checkpoint-path.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Optional ROI override as normalized floats (default matches training).",
    )
    parser.add_argument(
        "--local-video-path",
        type=Path,
        default=None,
        help="Optional path to a pre-downloaded video segment. Skips S3 download when provided.",
    )
    return parser.parse_args()


def download_video(s3_uri: str, dest_dir: Path, logger: logging.Logger) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    filename = Path(key).name
    if not filename:
        raise ValueError(f"S3 URI must reference a file: {s3_uri}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / filename
    logger.info("Downloading %s -> %s", s3_uri, local_path)
    get_s3_client().download_file(bucket, key, str(local_path))
    return local_path


def extract_segment_frames(
    video_path: Path,
    start_time: float,
    duration: float,
    logger: logging.Logger,
) -> tuple[List[np.ndarray], float]:
    if start_time < 0 or duration <= 0:
        raise ValueError("start_time must be >= 0 and duration > 0")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info("Video info -> fps: %.2f | frames: %d", fps, total_frames)

    start_idx = max(0, int(math.floor(start_time * fps)))
    end_idx = start_idx + int(math.ceil(duration * fps))
    if total_frames:
        end_idx = min(total_frames, end_idx)
    if end_idx <= start_idx:
        raise RuntimeError("Segment duration is too short; no frames selected")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames: List[np.ndarray] = []
    for idx in range(start_idx, end_idx):
        ret, frame = capture.read()
        if not ret:
            logger.warning("Stopped reading at frame %d (ret=%s)", idx, ret)
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()
    if not frames:
        raise RuntimeError("No frames decoded from selected segment")

    logger.info("Extracted %d frames for inference", len(frames))
    return frames, fps


def crop_roi(frame: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x = int(max(0, min(w - 1, roi[0] * w)))
    y = int(max(0, min(h - 1, roi[1] * h)))
    width = int(max(1, roi[2] * w))
    height = int(max(1, roi[3] * h))
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    return frame[y:y2, x:x2]


def prepare_windows(
    frames: List[np.ndarray],
    fps: float,
    roi_box: Sequence[float],
    logger: logging.Logger,
) -> List[Dict[str, float]]:
    roi_frames = [crop_roi(frame, roi_box) for frame in frames]
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        crop_size=constants.ENCODER_CONFIG["resolution"],
    )

    windows = []
    for start in range(0, len(roi_frames) - CLIP_FRAMES + 1, WINDOW_STRIDE):
        clip_frames = roi_frames[start : start + CLIP_FRAMES]
        tensor = transform(clip_frames)[0]
        windows.append(
            {
                "tensor": tensor,
                "start_idx": start,
                "end_idx": start + CLIP_FRAMES,
                "start_time": start / fps,
                "end_time": (start + CLIP_FRAMES) / fps,
            }
        )

    if not windows:
        raise RuntimeError(
            f"Not enough frames ({len(frames)}) for clip length {CLIP_FRAMES}; collect more video."
        )

    logger.info("Prepared %d sliding windows (stride=%d frames)", len(windows), WINDOW_STRIDE)
    return windows


def _load_checkpoint_state(path: Path, device: torch.device, logger: logging.Logger) -> Dict[str, torch.Tensor]:
    logger.debug("Loading checkpoint file %s", path)
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} does not contain a state_dict-compatible mapping")

    candidates = [
        payload.get("classifier"),
        payload.get("state_dict"),
        payload.get("pooler_state"),
    ]
    classifiers_entry = payload.get("classifiers")
    if isinstance(classifiers_entry, (list, tuple)) and classifiers_entry:
        candidates.append(classifiers_entry[0])

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    return payload


def _strip_prefixes(key: str) -> str:
    prefixes = ("module.", "classifier.", "attentive_classifier.")
    stripped = key
    prefix_applied = True
    while prefix_applied:
        prefix_applied = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                prefix_applied = True
    return stripped


def _extract_component_state(
    state_dict: Dict[str, torch.Tensor],
    component: str,
    reference_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    prefixes = ("pooler.",) if component == "pooler" else ("linear.", "head.")
    nested_candidates = (
        ("pooler_state", "pooler_weights", "pooler")
        if component == "pooler"
        else ("linear_head_state", "head_state", "linear_state", "linear_head")
    )

    for candidate in nested_candidates:
        nested = state_dict.get(candidate)
        if isinstance(nested, dict):
            state_dict = nested
            break

    extracted: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        normalized = _strip_prefixes(key)
        for prefix in prefixes:
            if normalized.startswith(prefix):
                trimmed = normalized[len(prefix) :]
                extracted[trimmed] = tensor
                break

    if not extracted:
        for key, tensor in state_dict.items():
            normalized = _strip_prefixes(key)
            if normalized in reference_keys:
                extracted[normalized] = tensor

    return extracted


def _log_load_warnings(
    logger: logging.Logger,
    label: str,
    load_result,
) -> None:
    missing = getattr(load_result, "missing_keys", None)
    unexpected = getattr(load_result, "unexpected_keys", None)
    if missing or unexpected:
        logger.warning("%s load discrepancies | missing=%s | unexpected=%s", label, missing, unexpected)


def load_models(
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path | None = None,
    pooler_checkpoint_path: Path | None = None,
    linear_head_checkpoint_path: Path | None = None,
):
    encoder = load_frozen_encoder(device, logger)
    freeze_pooler = constants.CLASSIFIER_CONFIG.get("freeze_pooler", False)
    logger.info("Building classifier | freeze_pooler=%s", freeze_pooler)
    classifier = build_classifier(encoder.embed_dim, device)
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info("Classifier params -> total: %d | trainable: %d", total_params, trainable_params)
    use_split = pooler_checkpoint_path is not None or linear_head_checkpoint_path is not None
    if use_split:
        if pooler_checkpoint_path is None or linear_head_checkpoint_path is None:
            raise ValueError("Both --pooler-checkpoint-path and --linear-head-checkpoint-path must be provided")
        if not pooler_checkpoint_path.exists():
            raise FileNotFoundError(f"Pooler checkpoint missing -> {pooler_checkpoint_path}")
        if not linear_head_checkpoint_path.exists():
            raise FileNotFoundError(f"Linear head checkpoint missing -> {linear_head_checkpoint_path}")

        logger.info(
            "Loading classifier from separate checkpoints | pooler=%s | linear_head=%s",
            pooler_checkpoint_path,
            linear_head_checkpoint_path,
        )
        pooler_state_raw = _load_checkpoint_state(pooler_checkpoint_path, device, logger)
        head_state_raw = _load_checkpoint_state(linear_head_checkpoint_path, device, logger)

        pooler_state = _extract_component_state(pooler_state_raw, "pooler", classifier.pooler.state_dict().keys())
        head_state = _extract_component_state(head_state_raw, "linear", classifier.linear.state_dict().keys())

        if not pooler_state:
            raise ValueError(f"No pooler weights found in {pooler_checkpoint_path}")
        if not head_state:
            raise ValueError(f"No linear head weights found in {linear_head_checkpoint_path}")

        pooler_result = classifier.pooler.load_state_dict(pooler_state, strict=False)
        _log_load_warnings(logger, "Pooler", pooler_result)

        head_result = classifier.linear.load_state_dict(head_state, strict=False)
        _log_load_warnings(logger, "Linear head", head_result)
        logger.info("Loaded classifier pooler/head from separate checkpoints")
    else:
        if checkpoint_path is None:
            raise ValueError("A classifier checkpoint path is required when separate pooler/head checkpoints are absent.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint missing -> {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        if "classifier" not in payload:
            raise KeyError(f"Classifier checkpoint {checkpoint_path} does not contain a 'classifier' entry")
        classifier.load_state_dict(payload["classifier"])
        logger.info(
            "Loaded classifier checkpoint from %s (epoch=%s, metric=%.2f)",
            checkpoint_path,
            payload.get("epoch"),
            payload.get("metric", 0.0),
        )

    classifier.eval()
    return encoder, classifier


def predict_windows(
    windows: List[Dict[str, float]],
    encoder,
    classifier,
    device: torch.device,
    logger: logging.Logger,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    encoder.eval()
    classifier.eval()

    def _logits(batch_tensor: torch.Tensor) -> torch.Tensor:
        clips = [[batch_tensor]]
        outputs = encoder(clips, clip_indices=None)
        return sum(classifier(o) for o in outputs) / len(outputs)

    with torch.inference_mode():
        for idx in range(0, len(windows), BATCH_SIZE):
            batch = windows[idx : idx + BATCH_SIZE]
            video_tensor = torch.stack([item["tensor"] for item in batch]).to(device)
            logits = _logits(video_tensor)
            probs = torch.softmax(logits, dim=1)
            for row, prob in zip(batch, probs):
                score, class_idx = torch.max(prob, dim=0)
                results.append(
                    {
                        "label": constants.CLASS_NAMES[int(class_idx)],
                        "prob": float(score),
                        "class_index": int(class_idx),
                        "start_idx": row["start_idx"],
                        "end_idx": row["end_idx"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                    }
                )

    logger.info("Generated predictions for %d windows", len(results))
    return results


def _per_frame_predictions(num_frames: int, predictions: List[Dict[str, float]]):
    frame_preds = [None] * num_frames
    for pred in predictions:
        for frame_idx in range(pred["start_idx"], min(num_frames, pred["end_idx"])):
            current = frame_preds[frame_idx]
            if current is None or pred["prob"] > current["prob"]:
                frame_preds[frame_idx] = pred
    return frame_preds


def annotate_and_save(
    frames: List[np.ndarray],
    predictions: List[Dict[str, float]],
    fps: float,
    source_path: Path,
    start_time: float,
    duration: float,
    roi_box: Sequence[float],
    logger: logging.Logger,
) -> tuple[Path, Path]:
    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_name = ANNOTATED_TEMPLATE.format(stem=source_path.stem, start=start_time, duration=duration)
    metadata_name = METADATA_TEMPLATE.format(stem=source_path.stem, start=start_time, duration=duration)
    video_path = OUTPUT_VIDEO_DIR / output_name
    metadata_path = OUTPUT_VIDEO_DIR / metadata_name

    height, width = frames[0].shape[:2]
    x = int(roi_box[0] * width)
    y = int(roi_box[1] * height)
    w_box = int(roi_box[2] * width)
    h_box = int(roi_box[3] * height)
    x2 = min(width - 1, x + w_box)
    y2 = min(height - 1, y + h_box)

    frame_preds = _per_frame_predictions(len(frames), predictions)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for idx, frame in enumerate(frames):
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 0), 3)
        pred = frame_preds[idx]
        if pred is not None:
            text = f"{pred['label']} ({pred['prob'] * 100:.1f}%)"
        else:
            text = "No prediction"
        cv2.putText(annotated, text, (x + 10, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    writer.release()
    with metadata_path.open("w") as fh:
        json.dump(predictions, fh, indent=2)

    logger.info("Annotated video saved to %s", video_path)
    return video_path, metadata_path


def upload_results(local_path: Path, metadata_path: Path, logger: logging.Logger):
    bucket, prefix = parse_s3_uri(RESULTS_S3_URI)
    prefix = prefix.rstrip("/")
    client = get_s3_client()
    video_key = f"{prefix}/{local_path.name}" if prefix else local_path.name
    meta_key = f"{prefix}/{metadata_path.name}" if prefix else metadata_path.name
    logger.info("Uploading annotated video to s3://%s/%s", bucket, video_key)
    client.upload_file(str(local_path), bucket, video_key)
    client.upload_file(str(metadata_path), bucket, meta_key)


def run_inference(
    s3_path: str,
    start_time: float,
    duration: float,
    checkpoint_path: Path | None = None,
    pooler_checkpoint_path: Path | None = None,
    linear_head_checkpoint_path: Path | None = None,
    roi_box: Sequence[float] = ROI_BOX,
    local_video_path: Path | None = None,
):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting inference | device=%s", device)

    use_split = pooler_checkpoint_path is not None or linear_head_checkpoint_path is not None
    if use_split and (pooler_checkpoint_path is None or linear_head_checkpoint_path is None):
        raise ValueError("Both --pooler-checkpoint-path and --linear-head-checkpoint-path must be provided together")

    if use_split:
        logger.info(
            "Using separate pooler/head checkpoints | pooler=%s | linear_head=%s",
            pooler_checkpoint_path,
            linear_head_checkpoint_path,
        )
        effective_checkpoint = None
    else:
        effective_checkpoint = checkpoint_path or CHECKPOINT_PATH
        if effective_checkpoint is None:
            raise ValueError("A classifier checkpoint path must be provided for inference.")
        logger.info("Using classifier checkpoint %s", effective_checkpoint)

    encoder, classifier = load_models(
        device,
        logger,
        checkpoint_path=effective_checkpoint,
        pooler_checkpoint_path=pooler_checkpoint_path,
        linear_head_checkpoint_path=linear_head_checkpoint_path,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if local_video_path:
            local_video = Path(local_video_path)
        else:
            local_video = download_video(s3_path, tmp_dir, logger)
        frames, fps = extract_segment_frames(local_video, start_time, duration, logger)
        windows = prepare_windows(frames, fps, roi_box, logger)
        predictions = predict_windows(windows, encoder, classifier, device, logger)
        video_path, metadata_path = annotate_and_save(
            frames,
            predictions,
            fps,
            local_video,
            start_time,
            duration,
            roi_box,
            logger,
        )
        upload_results(video_path, metadata_path, logger)

    logger.info("Inference complete. Uploaded results to %s", RESULTS_S3_URI)


def main():
    args = parse_args()
    roi_box = tuple(args.roi) if args.roi else ROI_BOX
    run_inference(
        args.s3_path,
        args.start_time,
        args.duration,
        args.checkpoint_path,
        args.pooler_checkpoint_path,
        args.linear_head_checkpoint_path,
        roi_box=roi_box,
        local_video_path=args.local_video_path,
    )


if __name__ == "__main__":
    main()
