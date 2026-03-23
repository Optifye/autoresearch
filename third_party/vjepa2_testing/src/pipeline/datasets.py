"""Dataset helpers for classifier training."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import decord
import torch
from torch.utils.data import Dataset

from evals.video_classification_frozen.utils import make_transforms

from src import constants


def _load_video_frames(path: Path) -> List:
    vr = decord.VideoReader(str(path))
    frames = []
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        if frame.ndim >= 3 and frame.shape[-1] > 3:
            frame = frame[..., :3]
        frames.append(frame)
    return frames


class VideoClipDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[Path, int]],
        transform: Callable[[List], List[torch.Tensor]],
        logger: logging.Logger,
    ) -> None:
        self.items = list(items)
        self.transform = transform
        self.logger = logger

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        frames = _load_video_frames(path)
        augmented = self.transform(frames)[0]  # shape: C x T x H x W
        return augmented, label


def _build_transforms(training: bool):
    disabled_ops = list(constants.RAND_AUGMENT_DISABLED_OPS)
    if not constants.AUGMENTATION_CONFIG.get("enable_movement_transforms", True):
        disabled_ops.extend(constants.MOVEMENT_AUTOAUGMENT_OPS)
    # Preserve order while removing duplicates
    disabled_ops = tuple(dict.fromkeys(disabled_ops))
    return make_transforms(
        training=training,
        random_horizontal_flip=False,
        num_views_per_clip=1,
        crop_size=constants.ENCODER_CONFIG["resolution"],
        auto_augment=training,
        autoaugment_disabled_ops=disabled_ops,
        randaugment_overrides=constants.RAND_AUGMENT_OP_OVERRIDES,
    )


def _split_paths(
    paths: Sequence[Path],
    train_split: float,
    rng: random.Random,
    class_name: str,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Path]]:
    items = list(paths)
    rng.shuffle(items)
    if not items:
        logger.warning("No validated clips for class '%s'", class_name)
        return [], []

    if len(items) == 1:
        logger.warning("Only one clip available for class '%s'; assigning to train set", class_name)
        return items, []

    split_idx = max(1, min(len(items) - 1, math.ceil(len(items) * train_split)))
    return items[:split_idx], items[split_idx:]


def create_datasets(
    class_to_paths: Dict[str, Sequence[Path]],
    train_split: float,
    seed: int,
    logger: logging.Logger,
    explicit_split: Dict[str, Dict[str, Sequence[Path]]] | None = None,
) -> Tuple[Dataset, Dataset, List[str]]:
    rng = random.Random(seed)
    train_items: List[Tuple[Path, int]] = []
    val_items: List[Tuple[Path, int]] = []
    label_names: List[str] = []
    per_class_stats = {}

    for label_idx, (name, paths) in enumerate(sorted(class_to_paths.items())):
        label_names.append(name)
        if explicit_split and name in explicit_split:
            train_paths = [Path(p) for p in explicit_split[name].get("train", [])]
            val_paths = [Path(p) for p in explicit_split[name].get("val", [])]
        else:
            train_paths, val_paths = _split_paths(paths, train_split, rng, name, logger)
        train_items.extend((p, label_idx) for p in train_paths)
        val_items.extend((p, label_idx) for p in val_paths)
        per_class_stats[name] = {"train": len(train_paths), "val": len(val_paths)}

    logger.info(
        "Dataset sizes -> train: %d, val: %d (total: %d)",
        len(train_items),
        len(val_items),
        len(train_items) + len(val_items),
    )
    for class_name, stats in per_class_stats.items():
        logger.info(
            "Class '%s' distribution -> train: %d | val: %d",
            class_name,
            stats["train"],
            stats["val"],
        )

    train_tfms = _build_transforms(training=True)
    val_tfms = _build_transforms(training=False)
    logger.info("Transforms configured | training=%s | validation=%s", train_tfms, val_tfms)
    logger.info(
        "Training transform flags | auto_augment=%s motion_shift=%s reprob=%s",
        getattr(train_tfms, "auto_augment", None),
        getattr(train_tfms, "motion_shift", None),
        getattr(train_tfms, "reprob", None),
    )

    train_dataset = VideoClipDataset(train_items, train_tfms, logger)
    val_dataset = VideoClipDataset(val_items, val_tfms, logger)

    return train_dataset, val_dataset, label_names
