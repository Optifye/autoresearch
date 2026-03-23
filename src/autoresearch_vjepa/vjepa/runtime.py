"""Minimal V-JEPA runtime for standalone dense-temporal extraction."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

import torch

LOGGER = logging.getLogger(__name__)


def prioritize_vendor_paths() -> None:
    """Ensure vendored V-JEPA modules shadow any other similarly named packages."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except IndexError:  # pragma: no cover
        return

    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    vendor_root = repo_root / "third_party" / "vjepa2_testing"
    vendor_paths = [
        vendor_root,
        vendor_root / "src",
        vendor_root / "vjepa2",
    ]
    inserted: List[str] = []
    for path in reversed(vendor_paths):
        if not path.exists():
            continue
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
        inserted.append(path_str)
    if inserted:
        LOGGER.debug("Prioritized V-JEPA vendor PYTHONPATH entries: %s", list(reversed(inserted)))


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class VJEPARuntimeConfig:
    device: str
    inference_dtype: str
    pooler_root: Path
    pooler_map_path: Optional[Path] = None
    vjepa_encoder_checkpoint: Optional[Path] = None


@dataclass(slots=True)
class VJEPABackboneRecord:
    model_name: str
    sha: str
    encoder: torch.nn.Module
    device: torch.device
    clip_len: int
    lock: RLock
    pooler: torch.nn.Module
    encoder_model: str
    encoder_checkpoint: Optional[str]
    embed_dim: Optional[int]
    resolution: int


class VJEPABackboneManager:
    def __init__(self, config: VJEPARuntimeConfig) -> None:
        self._config = config
        self._device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, VJEPABackboneRecord] = {}
        self._encoder_cache: Dict[tuple[str, Optional[str], int], tuple[torch.nn.Module, Optional[int], int]] = {}
        self._lock = RLock()
        self._pooler_root = Path(config.pooler_root).expanduser().resolve()
        self._pooler_map_path = (
            Path(config.pooler_map_path).expanduser().resolve() if config.pooler_map_path is not None else None
        )
        self._pooler_map = self._load_pooler_map()

    def get(self, model_name: Optional[str], *, clip_len: int) -> VJEPABackboneRecord:
        if not model_name:
            raise FileNotFoundError("V-JEPA pooler model_name (sha) is required")
        cache_key = f"{str(model_name)}|clip={int(clip_len)}"
        with self._lock:
            record = self._cache.get(cache_key)
            if record is None:
                pooler_path = self._pooler_map.get(str(model_name))
                if pooler_path is None:
                    raise FileNotFoundError(f"V-JEPA pooler '{model_name}' not available in local pooler map")
                record = self._load_record(str(model_name), int(clip_len), pooler_path)
                self._cache[cache_key] = record
            return record

    def _load_record(self, pooler_sha: str, clip_len: int, pooler_path: Path) -> VJEPABackboneRecord:
        LOGGER.info("Loading V-JEPA pooler '%s' (clip_len=%d)", pooler_sha, clip_len)
        pooler_payload = torch.load(pooler_path, map_location="cpu")
        metadata = pooler_payload.get("metadata", {}) if isinstance(pooler_payload, dict) else {}
        pooler_state = pooler_payload.get("pooler_state", pooler_payload) if isinstance(pooler_payload, dict) else pooler_payload
        meta_sha = metadata.get("pooler_sha")
        if meta_sha and str(meta_sha) != str(pooler_sha):
            raise RuntimeError(
                f"Pooler key {pooler_sha} does not match checkpoint metadata.pooler_sha {meta_sha} for {pooler_path}"
            )
        if isinstance(pooler_state, dict):
            pooler_state = {key: value for key, value in pooler_state.items() if not str(key).startswith("linear.")}
        encoder_model = str(metadata.get("encoder_model") or "large").strip().lower()
        resolved_ckpt = self._resolve_encoder_checkpoint(
            metadata.get("encoder_checkpoint"),
            encoder_model=encoder_model,
        )
        encoder, encoder_dim, resolution = self._get_or_build_encoder(
            encoder_model=encoder_model,
            encoder_checkpoint=resolved_ckpt,
            clip_len=int(clip_len),
        )
        classifier = self._build_classifier(int(getattr(encoder, "embed_dim", encoder_dim or 1024)))
        missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
        if missing or unexpected:
            LOGGER.warning("Pooler load warnings missing=%s unexpected=%s", missing[:8], unexpected[:8])
        return VJEPABackboneRecord(
            model_name=str(pooler_sha),
            sha=str(meta_sha or pooler_sha),
            encoder=encoder,
            device=self._device,
            clip_len=int(clip_len),
            lock=RLock(),
            pooler=classifier.pooler,
            encoder_model=encoder_model,
            encoder_checkpoint=(str(resolved_ckpt) if resolved_ckpt is not None else None),
            embed_dim=encoder_dim,
            resolution=int(resolution),
        )

    def _get_or_build_encoder(
        self,
        *,
        encoder_model: str,
        encoder_checkpoint: Optional[Path],
        clip_len: int,
    ) -> tuple[torch.nn.Module, Optional[int], int]:
        key = (str(encoder_model), str(encoder_checkpoint) if encoder_checkpoint else None, int(clip_len))
        cached = self._encoder_cache.get(key)
        if cached is not None:
            return cached
        encoder, resolution = self._build_encoder(
            encoder_model=str(encoder_model),
            encoder_checkpoint=encoder_checkpoint,
            clip_len=int(clip_len),
        )
        cached = (encoder, getattr(encoder, "embed_dim", None), int(resolution))
        self._encoder_cache[key] = cached
        return cached

    def _load_pooler_map(self) -> Dict[str, Path]:
        def _register(pooler_map: Dict[str, Path], pooler_path: Path, source: str) -> None:
            payload_obj = torch.load(pooler_path, map_location="cpu")
            if not isinstance(payload_obj, dict):
                raise RuntimeError(f"Failed to parse pooler checkpoint {pooler_path} ({source})")
            metadata = payload_obj.get("metadata", {})
            meta_sha = metadata.get("pooler_sha") if isinstance(metadata, dict) else None
            encoder_model = metadata.get("encoder_model") if isinstance(metadata, dict) else None
            if not meta_sha:
                raise RuntimeError(f"Pooler {pooler_path} missing metadata.pooler_sha")
            if not encoder_model:
                raise RuntimeError(f"Pooler {pooler_path} missing metadata.encoder_model")
            pooler_map[str(meta_sha)] = pooler_path

        pooler_map: Dict[str, Path] = {}
        if self._pooler_map_path is not None:
            if not self._pooler_map_path.exists():
                raise FileNotFoundError(f"V-JEPA pooler map not found: {self._pooler_map_path}")
            payload = json.loads(self._pooler_map_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload:
                raise FileNotFoundError(f"V-JEPA pooler map {self._pooler_map_path} is empty or invalid")
            for key, path_str in payload.items():
                pooler_path = Path(path_str).expanduser().resolve()
                if not pooler_path.exists():
                    raise FileNotFoundError(f"Pooler path {pooler_path} missing for sha {key}")
                _register(pooler_map, pooler_path, f"map {self._pooler_map_path}")
        else:
            if not self._pooler_root.exists():
                raise FileNotFoundError(f"V-JEPA pooler root {self._pooler_root} does not exist")
            candidates = list(self._pooler_root.rglob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No pooler checkpoints found under {self._pooler_root}")
            for path in candidates:
                _register(pooler_map, path, f"scan root {self._pooler_root}")
        if not pooler_map:
            raise FileNotFoundError("V-JEPA pooler map is empty after scan")
        return pooler_map

    def _resolve_encoder_checkpoint(
        self,
        encoder_ckpt: Optional[str],
        *,
        encoder_model: Optional[str] = None,
    ) -> Optional[Path]:
        if encoder_ckpt is None and self._config.vjepa_encoder_checkpoint is not None:
            candidate = Path(self._config.vjepa_encoder_checkpoint).expanduser()
            if candidate.exists():
                return candidate.resolve()
        if encoder_ckpt:
            candidate = Path(str(encoder_ckpt)).expanduser()
            if candidate.is_absolute() and candidate.exists():
                return candidate.resolve()

        try:
            repo_root = Path(__file__).resolve().parents[3]
        except IndexError:  # pragma: no cover
            repo_root = None
        workspace_root = os.getenv("AUTORESEARCH_WORKSPACE_ROOT", "").strip()
        roots = [
            (repo_root / "encoder_models") if repo_root else None,
            (Path(workspace_root).expanduser() / "encoder_models") if workspace_root else None,
            Path("/workspace/encoder_models"),
            Path("/app/encoder_models"),
            Path("/app/third_party/vjepa2_testing/encoder_models"),
        ]
        roots = [root for root in roots if root is not None]
        filenames: List[str] = []
        target = (encoder_model or "").strip().lower()
        if target == "large":
            filenames.append("vitl.pt")
        elif target == "giant":
            filenames.append("vitg-384.pt")
        else:
            filenames.extend(["vitg-384.pt", "vitl.pt"])
        if encoder_ckpt:
            filenames.insert(0, str(encoder_ckpt))

        for root in roots:
            for name in filenames:
                candidate = Path(name)
                path = candidate if candidate.is_absolute() else root / candidate
                if path.exists():
                    return path.resolve()
        LOGGER.warning("Unable to resolve encoder checkpoint '%s'; falling back to vendor defaults.", encoder_ckpt or target)
        return None

    def _build_encoder(
        self,
        *,
        encoder_model: str,
        encoder_checkpoint: Optional[Path] = None,
        clip_len: int,
    ) -> tuple[torch.nn.Module, int]:
        target_model = (encoder_model or "large").strip().lower()
        if target_model not in {"large", "giant"}:
            raise ValueError(f"Unsupported V-JEPA encoder model '{encoder_model}'")
        resolved_ckpt = self._resolve_encoder_checkpoint(
            str(encoder_checkpoint) if encoder_checkpoint is not None else None,
            encoder_model=target_model,
        )
        env_overrides: Dict[str, Optional[str]] = {
            "VJEPA_ENCODER_MODEL": target_model,
            "VJEPA_ENCODER_CHECKPOINT": str(resolved_ckpt) if resolved_ckpt is not None else None,
            "VJEPA_NUM_FRAMES": str(int(max(1, clip_len))),
            "VJEPA_CLIP_LEN": str(int(max(1, clip_len))),
        }
        prioritize_vendor_paths()
        with _temporary_env(env_overrides):
            from third_party.vjepa2_testing.src import constants as vjepa_constants
            from third_party.vjepa2_testing.src.pipeline.model_utils import load_frozen_encoder

            variant = vjepa_constants.ALLOWED_ENCODERS[target_model]
            checkpoint_path = resolved_ckpt if resolved_ckpt is not None else Path(vjepa_constants.SELECTED_ENCODER.checkpoint)
            selected_variant = vjepa_constants.EncoderVariant(
                key=variant.key,
                model_name=variant.model_name,
                checkpoint=Path(checkpoint_path),
                checkpoint_key=str(variant.checkpoint_key),
                embed_dim=variant.embed_dim,
                num_heads=variant.num_heads,
                depth=variant.depth,
                resolution=variant.resolution,
            )
            vjepa_constants.SELECTED_ENCODER = selected_variant
            vjepa_constants.ENCODER_CONFIG["model_kwargs"]["encoder"]["model_name"] = variant.model_name
            vjepa_constants.ENCODER_CONFIG["model_kwargs"]["encoder"]["checkpoint_key"] = str(variant.checkpoint_key)
            vjepa_constants.ENCODER_CONFIG["resolution"] = int(variant.resolution)
            vjepa_constants.ENCODER_CONFIG["checkpoint"] = str(selected_variant.checkpoint)
            vjepa_constants.NUM_FRAMES_IN_CLIP = int(max(1, clip_len))
            vjepa_constants.ENCODER_CONFIG["frames_per_clip"] = int(max(1, clip_len))
            wrapper_kwargs = dict(vjepa_constants.ENCODER_CONFIG.get("wrapper_kwargs", {}) or {})
            wrapper_kwargs["max_frames"] = max(int(wrapper_kwargs.get("max_frames", clip_len)), int(clip_len))
            vjepa_constants.ENCODER_CONFIG["wrapper_kwargs"] = wrapper_kwargs
            encoder = load_frozen_encoder(self._device, LOGGER, checkpoint=str(selected_variant.checkpoint))

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder, int(selected_variant.resolution)

    def _build_classifier(self, embed_dim: int):
        prioritize_vendor_paths()
        from third_party.vjepa2_testing.src.pipeline.model_utils import build_classifier

        classifier = build_classifier(int(embed_dim), self._device)
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        return classifier
