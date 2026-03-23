"""Pipeline constants for custom classifier training."""

from pathlib import Path
import json
import os
import sys
from dataclasses import dataclass


# Mapping from class label to one or more S3 prefixes containing the clips for that class.
# Multiple folders can be specified per class; defaults below keep the single-path behavior.
S3_CLASS_PATHS = {
    "continue": [
        "s3://optiframe-models/3ca708c4-c020-47c1-8993-7f3cf0edc6f1/14/clips/continue_with_roi/", # join shoulder
        "s3://optiframe-models/97014373-264b-46ef-b868-510a8d7f1dfd/7/clips/continue_with_roi/", # hem bottom
        "s3://optiframe-models/13f48d50-c6d6-4f46-aaee-29b491c808b9/18/clips/continue_with_roi/", # top stich
        "s3://optiframe-models/189fec81-10de-4c85-a489-e5e90877bfe5/27/clips/continue_with_roi/", # set collor
        "s3://optiframe-models/d4d6712e-b8bd-49f5-b09e-2d1a4093cb62/22/clips/continue_with_roi/", # closestich
    ],
    "end": [
        "s3://optiframe-models/3ca708c4-c020-47c1-8993-7f3cf0edc6f1/14/clips/end_with_roi/",
        "s3://optiframe-models/97014373-264b-46ef-b868-510a8d7f1dfd/7/clips/end_with_roi/",
        "s3://optiframe-models/13f48d50-c6d6-4f46-aaee-29b491c808b9/18/clips/end_with_roi/",
        "s3://optiframe-models/189fec81-10de-4c85-a489-e5e90877bfe5/27/clips/end_with_roi/",
        "s3://optiframe-models/d4d6712e-b8bd-49f5-b09e-2d1a4093cb62/22/clips/end_with_roi/",
    ],
    "no_action": [
        "s3://optiframe-models/3ca708c4-c020-47c1-8993-7f3cf0edc6f1/14/clips/no_action_with_roi/",  
        "s3://optiframe-models/97014373-264b-46ef-b868-510a8d7f1dfd/7/clips/no_action_with_roi/",
        "s3://optiframe-models/13f48d50-c6d6-4f46-aaee-29b491c808b9/18/clips/no_action_with_roi/",
        "s3://optiframe-models/189fec81-10de-4c85-a489-e5e90877bfe5/27/clips/no_action_with_roi/",
        "s3://optiframe-models/d4d6712e-b8bd-49f5-b09e-2d1a4093cb62/22/clips/no_action_with_roi/",
    ],
    "start": [
        "s3://optiframe-models/3ca708c4-c020-47c1-8993-7f3cf0edc6f1/14/clips/start_with_roi/",
        "s3://optiframe-models/97014373-264b-46ef-b868-510a8d7f1dfd/7/clips/start_with_roi/",
        "s3://optiframe-models/13f48d50-c6d6-4f46-aaee-29b491c808b9/18/clips/start_with_roi/",
        "s3://optiframe-models/189fec81-10de-4c85-a489-e5e90877bfe5/27/clips/start_with_roi/",
        "s3://optiframe-models/d4d6712e-b8bd-49f5-b09e-2d1a4093cb62/22/clips/start_with_roi/",
    ],
}

CLASS_NAMES = list(S3_CLASS_PATHS.keys())
NUM_CLASSES = len(CLASS_NAMES)


# Paths
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.getenv("VJEPA_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT))
DATA_ROOT = PROJECT_ROOT / "data"
RAW_CLIP_DIR = DATA_ROOT / "raw"
VALIDATED_CLIP_DIR = DATA_ROOT / "validated"
CLASSIFIER_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "classifiers"
DEFAULT_CLASSIFIER_SUBDIR = "default_run"
CHECKPOINT_DIR = CLASSIFIER_ARTIFACT_DIR / DEFAULT_CLASSIFIER_SUBDIR
POOLER_CHECKPOINT_PATH = CHECKPOINT_DIR / "pooler_pretrained.pt"
LINEAR_HEAD_CHECKPOINT_PATH = CHECKPOINT_DIR / "linear_head_latest.pt"
CHECKPOINT_PATH = LINEAR_HEAD_CHECKPOINT_PATH


# Data + preprocessing configuration
NUM_FRAMES_IN_CLIP = 4
_clip_len_override = os.getenv("VJEPA_NUM_FRAMES") or os.getenv("VJEPA_CLIP_LEN")
if _clip_len_override:
    try:
        NUM_FRAMES_IN_CLIP = max(1, int(_clip_len_override))
    except ValueError:
        pass
TRAIN_SPLIT = 0.8
RANDOM_SEED = 1337


# Training configuration
BATCH_SIZE = int(os.getenv("VJEPA_BATCH_SIZE", "8"))
NUM_EPOCHS = int(os.getenv("VJEPA_EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("VJEPA_LR", os.getenv("VJEPA_LEARNING_RATE", "1e-3")))
WEIGHT_DECAY = float(os.getenv("VJEPA_WEIGHT_DECAY", "0.01"))
NUM_WORKERS = 4
PIN_MEMORY = True
USE_BFLOAT16 = True


# Logging
LOG_LEVEL = "INFO"
LOG_FILENAME = CHECKPOINT_DIR / "training.log"

@dataclass(frozen=True)
class EncoderVariant:
    key: str
    model_name: str
    checkpoint: Path
    checkpoint_key: str
    embed_dim: int
    num_heads: int
    depth: int
    resolution: int


ALLOWED_ENCODERS = {
    "giant": EncoderVariant(
        key="giant",
        model_name="vit_giant_xformers",
        checkpoint=Path("encoder_models/vitg-384.pt"),
        checkpoint_key="target_encoder",
        embed_dim=1408,
        num_heads=22,
        depth=40,
        resolution=384,
    ),
    "large": EncoderVariant(
        key="large",
        model_name="vit_large",
        checkpoint=Path("encoder_models/vitl.pt"),
        checkpoint_key="target_encoder",
        embed_dim=1024,
        num_heads=16,
        depth=24,
        resolution=256,
    ),
}


def _resolve_encoder_variant() -> EncoderVariant:
    name = (os.getenv("VJEPA_ENCODER_MODEL") or "giant").strip().lower()
    if name not in ALLOWED_ENCODERS:
        raise ValueError(
            f"Unknown VJEPA_ENCODER_MODEL '{name}'. Allowed values: {', '.join(sorted(ALLOWED_ENCODERS))}."
        )
    variant = ALLOWED_ENCODERS[name]
    ckpt_env = os.getenv("VJEPA_ENCODER_CHECKPOINT")
    ckpt_path = Path(ckpt_env) if ckpt_env else variant.checkpoint
    candidate_paths = []
    if ckpt_path.is_absolute():
        candidate_paths.append(ckpt_path.expanduser())
    else:
        for base in (
            PROJECT_ROOT,
            _DEFAULT_PROJECT_ROOT,
            Path(__file__).resolve().parents[2],  # vendor root
            Path(__file__).resolve().parents[4],  # repo root
        ):
            candidate_paths.append((base / ckpt_path).expanduser())
    resolved_ckpt = next((p.resolve() for p in candidate_paths if p.exists()), None)
    if resolved_ckpt is None:
        raise FileNotFoundError(
            f"Encoder checkpoint not found at any of: {', '.join(str(p) for p in candidate_paths)} "
            f"(variant={variant.model_name}). Set VJEPA_ENCODER_CHECKPOINT to the correct path."
        )
    checkpoint_key = os.getenv("VJEPA_ENCODER_CHECKPOINT_KEY") or variant.checkpoint_key
    resolved = EncoderVariant(
        key=variant.key,
        model_name=variant.model_name,
        checkpoint=resolved_ckpt,
        checkpoint_key=checkpoint_key,
        embed_dim=variant.embed_dim,
        num_heads=variant.num_heads,
        depth=variant.depth,
        resolution=variant.resolution,
    )
    print(
        f"[VJEPA] Encoder variant: {resolved.key} ({resolved.model_name}) | "
        f"ckpt={resolved.checkpoint} key={resolved.checkpoint_key} | "
        f"expected_dim={resolved.embed_dim} heads={resolved.num_heads} depth={resolved.depth} | "
        f"resolution={resolved.resolution} frames={NUM_FRAMES_IN_CLIP}",
        file=sys.stderr,
    )
    return resolved


SELECTED_ENCODER = _resolve_encoder_variant()


# Frozen encoder configuration (defaults can be overridden before running the pipeline)
ENCODER_CONFIG = {
    "module_name": "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip",
    "checkpoint": str(SELECTED_ENCODER.checkpoint),
    "resolution": SELECTED_ENCODER.resolution,
    "frames_per_clip": NUM_FRAMES_IN_CLIP,
    "model_kwargs": {
        "encoder": {
            "checkpoint_key": SELECTED_ENCODER.checkpoint_key,
            "model_name": SELECTED_ENCODER.model_name,
            "patch_size": 16,
            "tubelet_size": 2,
            "uniform_power": True,
            "use_rope": True,
        }
    },
    "wrapper_kwargs": {
        "max_frames": 128,
        "use_pos_embed": False,
    },
}
ENCODER_CONFIG["wrapper_kwargs"]["max_frames"] = max(
    ENCODER_CONFIG["wrapper_kwargs"].get("max_frames", NUM_FRAMES_IN_CLIP),
    NUM_FRAMES_IN_CLIP,
)


# Classifier configuration
CLASSIFIER_CONFIG = {
    "num_heads": 16,
    "num_probe_blocks": 4,
    "freeze_pooler": False,
    "pooler_checkpoint": str(POOLER_CHECKPOINT_PATH),
    "head_checkpoint": str(LINEAR_HEAD_CHECKPOINT_PATH),
}


def _load_json_env(path_or_payload: str) -> dict:
    candidate = Path(path_or_payload)
    if candidate.exists():
        return json.loads(candidate.read_text())
    return json.loads(path_or_payload)


_class_paths_override = os.getenv("VJEPA_CLASS_PATHS_JSON") or os.getenv("VJEPA_CLASS_PATHS")
if _class_paths_override:
    S3_CLASS_PATHS = _load_json_env(_class_paths_override)
    CLASS_NAMES = list(S3_CLASS_PATHS.keys())
    NUM_CLASSES = len(CLASS_NAMES)

_data_root_override = os.getenv("VJEPA_DATA_ROOT")
if _data_root_override:
    DATA_ROOT = Path(_data_root_override)
    RAW_CLIP_DIR = DATA_ROOT / "raw"
    VALIDATED_CLIP_DIR = DATA_ROOT / "validated"

_artifacts_override = os.getenv("VJEPA_CLASSIFIER_ARTIFACTS")
if _artifacts_override:
    CLASSIFIER_ARTIFACT_DIR = Path(_artifacts_override)
    DEFAULT_CLASSIFIER_SUBDIR = os.getenv("VJEPA_CHECKPOINT_SUBDIR", DEFAULT_CLASSIFIER_SUBDIR)
    CHECKPOINT_DIR = CLASSIFIER_ARTIFACT_DIR / DEFAULT_CLASSIFIER_SUBDIR

_checkpoint_subdir_override = os.getenv("VJEPA_CHECKPOINT_SUBDIR")
if _checkpoint_subdir_override:
    CHECKPOINT_DIR = CLASSIFIER_ARTIFACT_DIR / _checkpoint_subdir_override

POOLER_CHECKPOINT_PATH = Path(os.getenv("VJEPA_POOLER_CHECKPOINT", POOLER_CHECKPOINT_PATH))
LINEAR_HEAD_CHECKPOINT_PATH = CHECKPOINT_DIR / "linear_head_latest.pt"
CHECKPOINT_PATH = LINEAR_HEAD_CHECKPOINT_PATH

CLASSIFIER_CONFIG["pooler_checkpoint"] = str(POOLER_CHECKPOINT_PATH)
CLASSIFIER_CONFIG["head_checkpoint"] = str(LINEAR_HEAD_CHECKPOINT_PATH)

_freeze_pooler_override = os.getenv("VJEPA_FREEZE_POOLER")
if _freeze_pooler_override is not None:
    CLASSIFIER_CONFIG["freeze_pooler"] = _freeze_pooler_override.lower() in {"1", "true", "yes"}

_log_filename = os.getenv("VJEPA_LOG_FILENAME")
if _log_filename:
    LOG_FILENAME = Path(_log_filename)


# Augmentation configuration
RAND_AUGMENT_OP_OVERRIDES = {
    # Translation magnitudes are relative percentages of width/height.
    "TranslateXRel": {"translate_pct": 0.1},
    "TranslateYRel": {"translate_pct": 0.1},
    # Keep SolarizeAdd active but cap the added intensity.
    "SolarizeAdd": {"solarize_add_max": 60},
    # Disable SolarizeIncreasing entirely.
    "SolarizeIncreasing": {"enabled": False},
    # Shear magnitude expressed as fractional skew.
    "ShearX": {"max_shear": 0.07},
    "ShearY": {"max_shear": 0.07},
    # Limit rotation range (degrees).
    "Rotate": {"max_rotate_degrees": 5},
    # Reduce probability of applying Invert.
    "Invert": {"prob": 0.25},
    # Cap brightness delta so 1.45x is the maximum boost.
    "BrightnessIncreasing": {"max_enhance_increase": 0.45},
}


def _derive_disabled_ops(overrides):
    return tuple(name for name, cfg in overrides.items() if cfg.get("enabled") is False)


RAND_AUGMENT_DISABLED_OPS = _derive_disabled_ops(RAND_AUGMENT_OP_OVERRIDES)
BASE_AUTOAUGMENT_DISABLED_OPS = RAND_AUGMENT_DISABLED_OPS

MOVEMENT_AUTOAUGMENT_OPS = ("Rotate", "ShearX", "ShearY", "TranslateYRel")
AUGMENTATION_CONFIG = {
    # Movement ops stay enabled unless explicitly disabled in a config file.
    "enable_movement_transforms": True,
}
# Ensure imports via both `constants` and `src.constants` receive the same module instance.
_module = sys.modules[__name__]
sys.modules["constants"] = _module
sys.modules["src.constants"] = _module
