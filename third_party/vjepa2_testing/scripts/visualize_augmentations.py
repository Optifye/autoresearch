#!/usr/bin/env python3
"""Visualize every augmentation used by the training pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torchvision import transforms as tv_transforms

# Ensure the repo root and upstream V-JEPA package are importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VJEPA_ROOT = REPO_ROOT / "vjepa2"
if VJEPA_ROOT.exists() and str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))

try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None

from src import constants
from evals.video_classification_frozen.utils import VideoTransform, tensor_normalize

import src.datasets.utils.video.transforms as video_transforms
from vjepa2.src.datasets.utils.video import randaugment as ra_mod


# Matches the policy hard-coded inside VideoTransform.
AUTOAUGMENT_POLICY = "rand-m7-n4-mstd0.5-inc1"


@contextmanager
def temporary_rng_seed(seed: int | None):
    """Context manager that restores RNG state after running."""

    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the source video. Accepts local paths or s3://bucket/key URIs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "augmentation_viz",
        help="Directory where visualizations will be written.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample evenly from the video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base seed for deterministic augmentation sampling.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also render RandAugment transforms that are disabled in training.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on frames decoded from the video (defaults to clip length).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def fetch_video(video_uri: str, scratch_dir: Path) -> Path:
    """Download the source video (S3) or resolve local path."""

    if video_uri.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("boto3 is required to download from S3; run `pip install boto3`.")
        bucket, key = parse_s3_uri(video_uri)
        local_path = scratch_dir / Path(key).name
        client = boto3.client("s3")
        print(f"Downloading {video_uri} -> {local_path}")
        client.download_file(bucket, key, str(local_path))
        return local_path

    local_path = Path(video_uri).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_uri}")
    return local_path


def sample_frames(video_path: Path, num_frames: int, max_frames: int | None = None) -> Tuple[List[np.ndarray], List[int]]:
    """Evenly sample frames from the given video."""

    vr = VideoReader(str(video_path))
    total_frames = len(vr)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    frame_list = [frames[i] for i in range(len(indices))]
    return frame_list, indices


def numpy_to_pil(array: np.ndarray, size: Tuple[int, int] | None = None) -> Image.Image:
    img = Image.fromarray(array.astype(np.uint8))
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def save_frame_sequence(frames: Sequence[np.ndarray], out_dir: Path, prefix: str) -> List[str]:
    ensure_dir(out_dir)
    written = []
    for idx, frame in enumerate(frames):
        img = numpy_to_pil(frame)
        out_path = out_dir / f"{prefix}_{idx:02d}.png"
        img.save(out_path)
        written.append(str(out_path))
    return written


def build_comparison_grid(
    reference: Sequence[np.ndarray],
    augmented: Sequence[np.ndarray],
    display_size: Tuple[int, int],
) -> Image.Image:
    """Create a two-row grid showing original vs. augmented frames."""

    pad = 8
    cols = len(reference)
    width = cols * display_size[0] + (cols + 1) * pad
    rows = 2
    height = rows * display_size[1] + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    for col, (orig, aug) in enumerate(zip(reference, augmented)):
        orig_img = numpy_to_pil(orig, display_size)
        aug_img = numpy_to_pil(aug, display_size)
        x = pad + col * (display_size[0] + pad)
        canvas.paste(orig_img, (x, pad))
        canvas.paste(aug_img, (x, pad * 2 + display_size[1]))
    return canvas


def infer_op_name(op: ra_mod.AugmentOp) -> str:
    """Map an AugmentOp back to its registered name."""

    for name, fn in ra_mod.NAME_TO_OP.items():
        if fn is op.aug_fn and ra_mod.LEVEL_TO_ARG[name] is op.level_fn:
            return name
    raise RuntimeError("Unable to infer RandAugment op name.")


class TrackingRandAugment(ra_mod.RandAugment):
    """RandAugment variant that records which ops were applied."""

    def __init__(self, base: ra_mod.RandAugment, name_lookup: Dict[int, str]):
        super().__init__(base.ops, base.num_layers, base.choice_weights)
        self._name_lookup = name_lookup
        self.history: List[List[str]] = []

    def __call__(self, img):
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        applied_names = []
        for op in ops:
            applied_names.append(self._name_lookup.get(id(op), "unknown"))
            img = op(img)
        self.history.append(applied_names)
        return img


class AugmentationVisualizer:
    def __init__(
        self,
        seed: int,
        output_dir: Path,
        include_disabled: bool,
    ) -> None:
        self.seed = seed
        self.output_dir = ensure_dir(output_dir)
        self.include_disabled = include_disabled
        self.normalize = (
            list(constants.ENCODER_CONFIG.get("normalize_mean", (0.485, 0.456, 0.406))),
            list(constants.ENCODER_CONFIG.get("normalize_std", (0.229, 0.224, 0.225))),
        )
        self.crop_size = constants.ENCODER_CONFIG["resolution"]
        self.disabled_ops = constants.RAND_AUGMENT_DISABLED_OPS
        self.op_overrides = constants.RAND_AUGMENT_OP_OVERRIDES
        self.video_transform = VideoTransform(
            training=True,
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(3 / 4, 4 / 3),
            random_resize_scale=(0.3, 1.0),
            reprob=0.0,
            auto_augment=True,
            motion_shift=False,
            crop_size=self.crop_size,
            normalize=self.normalize,
            autoaugment_disabled_ops=self.disabled_ops,
            randaugment_overrides=self.op_overrides,
        )
        self.randaugment_tracker = self._wrap_randaugment()
        self.full_randaugment_ops = self._build_full_randaugment_op_map()
        self.to_pil = tv_transforms.ToPILImage()
        self.to_tensor = tv_transforms.ToTensor()

    def _wrap_randaugment(self) -> TrackingRandAugment | None:
        if not getattr(self.video_transform, "auto_augment", False):
            return None
        comp = self.video_transform.autoaug_transform
        transforms_list: Iterable = getattr(comp, "transforms", [comp])
        for idx, transform in enumerate(transforms_list):
            if isinstance(transform, ra_mod.RandAugment):
                name_lookup = {id(op): infer_op_name(op) for op in transform.ops}
                tracker = TrackingRandAugment(transform, name_lookup)
                if hasattr(comp, "transforms"):
                    comp.transforms[idx] = tracker
                else:  # pragma: no cover - defensive fallback
                    self.video_transform.autoaug_transform = tracker
                return tracker
        return None

    def _build_full_randaugment_op_map(self) -> Dict[str, ra_mod.AugmentOp]:
        """Instantiate all RandAugment ops (without disabling) for reuse."""

        # Try to reuse the same helper as the training transform; if that fails, fall back to
        # constructing the RandAugment instance directly.
        compose = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=AUTOAUGMENT_POLICY,
            interpolation="bicubic",
            disabled_ops=None,
            op_overrides=self.op_overrides,
        )
        transforms_list: Iterable = getattr(compose, "transforms", [compose])
        for transform in transforms_list:
            if isinstance(transform, ra_mod.RandAugment):
                return {infer_op_name(op): op for op in transform.ops}

        # Fallback path: instantiate a fresh RandAugment object using the same hyper-parameters.
        aa_params = {"translate_const": int(self.crop_size * 0.45)}
        get_interp = getattr(video_transforms, "_pil_interp", None)
        if callable(get_interp):
            aa_params["interpolation"] = get_interp("bicubic")
        randaug = ra_mod.rand_augment_transform(
            AUTOAUGMENT_POLICY,
            aa_params,
            disabled_ops=None,
            op_overrides=self.op_overrides,
        )
        if isinstance(randaug, ra_mod.RandAugment):
            return {infer_op_name(op): op for op in randaug.ops}
        raise RuntimeError("Unable to locate RandAugment instance in transform pipeline.")

    def _clone_aug_op(self, name: str, force_prob: float = 1.0) -> ra_mod.AugmentOp:
        template = self.full_randaugment_ops[name]
        return ra_mod.AugmentOp(name, prob=force_prob, magnitude=template.magnitude, hparams=template.hparams)

    def visualize_randaugment_ops(self, frames: Sequence[np.ndarray]) -> Dict[str, Dict[str, str]]:
        """Apply each RandAugment op individually and store comparison grids."""

        base_frames = [frame.copy() for frame in frames]
        names = sorted(self.full_randaugment_ops.keys())
        if not self.include_disabled:
            names = [n for n in names if n not in self.disabled_ops]

        results = {}
        for idx, name in enumerate(names):
            op = self._clone_aug_op(name, force_prob=1.0)
            with temporary_rng_seed(self.seed + idx):
                augmented = op([self.to_pil(frame) for frame in base_frames])
            augmented_np = [np.array(img) for img in augmented]
            status = "enabled" if name not in self.disabled_ops else "disabled"
            stage_dir = ensure_dir(self.output_dir / "randaugment_ops" / status)
            grid = build_comparison_grid(base_frames, augmented_np, (self.crop_size, self.crop_size))
            out_path = stage_dir / f"{idx:02d}_{name}.png"
            grid.save(out_path)
            results[name] = {"status": status, "file": str(out_path.relative_to(self.output_dir))}
        return results

    def run_training_pipeline(self, frames: Sequence[np.ndarray]) -> Dict[str, Sequence[np.ndarray] | List[str]]:
        """Replicates VideoTransform.__call__ with instrumentation."""

        pil_frames = [self.to_pil(frame) for frame in frames]
        with temporary_rng_seed(self.seed):
            aug_frames = pil_frames
            if getattr(self.video_transform, "auto_augment", False):
                aug_frames = self.video_transform.autoaug_transform([img.copy() for img in pil_frames])

            aug_arrays = [np.array(img) for img in aug_frames]
            tensor_frames = [self.to_tensor(img) for img in aug_frames]
            buffer = torch.stack(tensor_frames)  # T x C x H x W
            buffer = buffer.permute(0, 2, 3, 1)  # T x H x W x C

            buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
            buffer = buffer.permute(3, 0, 1, 2)  # C x T x H x W

            buffer = self.video_transform.spatial_transform(
                images=buffer,
                target_height=self.crop_size,
                target_width=self.crop_size,
                scale=self.video_transform.random_resize_scale,
                ratio=self.video_transform.random_resize_aspect_ratio,
            )

            did_flip = False
            if self.video_transform.random_horizontal_flip:
                buffer, _ = video_transforms.horizontal_flip(0.5, buffer)
                did_flip = True

            if self.video_transform.reprob > 0:
                buffer = buffer.permute(1, 0, 2, 3)
                buffer = self.video_transform.erase_transform(buffer)
                buffer = buffer.permute(1, 0, 2, 3)

        frames_after = clip_tensor_to_uint8(buffer, self.normalize)
        return {
            "randaugment_frames": aug_arrays,
            "final_frames": frames_after,
            "horizontal_flip": did_flip,
            "tensor": buffer,
        }


def clip_tensor_to_uint8(tensor: torch.Tensor, normalize: Tuple[Sequence[float], Sequence[float]]) -> List[np.ndarray]:
    """Denormalize and convert C x T x H x W tensors back to uint8 images."""

    mean = torch.tensor(normalize[0]).view(-1, 1, 1, 1)
    std = torch.tensor(normalize[1]).view(-1, 1, 1, 1)
    denorm = tensor * std + mean
    denorm = denorm.clamp(0.0, 1.0)
    frames = denorm.permute(1, 2, 3, 0).mul(255.0).byte().cpu().numpy()
    return [frame for frame in frames]


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)

    ensure_dir(args.output_dir)
    metadata = {
        "video": args.video,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "disabled_ops": list(constants.RAND_AUGMENT_DISABLED_OPS),
        "op_overrides": constants.RAND_AUGMENT_OP_OVERRIDES,
        "autoaugment_policy": AUTOAUGMENT_POLICY,
        "outputs": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        local_video = fetch_video(args.video, Path(tmpdir))
        frames, indices = sample_frames(local_video, args.num_frames, args.max_frames)
        metadata["frame_indices"] = indices

        viz = AugmentationVisualizer(
            seed=args.seed,
            output_dir=args.output_dir,
            include_disabled=args.include_disabled,
        )

        # Save raw frames for reference.
        raw_dir = ensure_dir(args.output_dir / "raw_frames")
        metadata["outputs"]["raw_frames"] = save_frame_sequence(frames, raw_dir, "raw")

        # Individual RandAugment ops.
        metadata["outputs"]["randaugment_ops"] = viz.visualize_randaugment_ops(frames)

        # Full pipeline pass.
        pipeline_outputs = viz.run_training_pipeline(frames)
        ra_dir = ensure_dir(args.output_dir / "randaugment_sequence")
        metadata["outputs"]["randaugment_sequence"] = save_frame_sequence(
            pipeline_outputs["randaugment_frames"],
            ra_dir,
            "randaug",
        )

        final_dir = ensure_dir(args.output_dir / "training_output")
        metadata["outputs"]["full_pipeline"] = save_frame_sequence(
            pipeline_outputs["final_frames"],
            final_dir,
            "final",
        )

        if viz.randaugment_tracker:
            metadata["randaugment_history"] = viz.randaugment_tracker.history

    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Visualization complete -> {args.output_dir}")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
