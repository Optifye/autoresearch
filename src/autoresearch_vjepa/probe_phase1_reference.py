"""Probe-only (phase-1) finetuning against a frozen boundary TCN."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .feature_store import FeatureStore, open_feature_store
from .losses import bce_with_logits, masked_mean
from .tcn_train import (
    _boundary_loss_with_logits,
    _cycle_ce_loss,
    _masked_multiclass_ce,
    _phase_loss_masked,
    _prepare_supervised_stream,
    _ranking_loss_start_end,
    _resolve_cycle_pos_weight,
    _select_chunk_start,
    _visible_cycles_in_chunk,
    load_boundary_checkpoint,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbePhase1Config:
    epochs: int
    probe_lr: float
    weight_decay: float
    grad_clip_norm: float
    chunk_len: int
    chunks_per_stream: int
    neg_chunk_fraction: float
    stage1_last_block_epoch: int
    stage1_all_epoch: int
    tcn_tune_mode: str = "frozen"
    tcn_last_blocks: int = 1
    tcn_lr: float = 5e-5
    tcn_weight_decay: float = 1e-4
    stream_sampling_mode: str = "uniform"
    stream_sampling_power: float = 0.5
    stream_sampling_min_weight: float = 1.0
    cyclece_weight: float = 1.0
    cyclece_tau: float = 0.7
    cyclece_radius: int = 1
    smooth_weight: float = 0.02
    distill_weight: float = 0.10
    class_weight: float = 1.0
    fail_if_best_epoch_zero: bool = False
    boundary_index_mode: str = "nearest"
    temporal_structure_mode: str = "cyclic"
    seed: int = 42


@dataclass(frozen=True)
class ProbePhase1Result:
    probe_checkpoint: Path
    tcn_checkpoint: Path
    metrics_path: Path
    best_epoch: int
    best_loss: float


@dataclass(frozen=True)
class _StageStream:
    name: str
    tokens: np.ndarray  # [T,N,D]
    z0: np.ndarray  # [T,D]
    y_start: np.ndarray
    y_end: np.ndarray
    y_boundary: np.ndarray
    y_cycle: np.ndarray
    y_phase: np.ndarray
    y_class_label_id: np.ndarray
    mask_start_end: np.ndarray
    mask_cycle: np.ndarray
    mask_phase: np.ndarray
    mask_class: np.ndarray
    mapped_cycles_idx: List[Tuple[int, int]]
    boundary_indices: List[int]
    supervised_mode: str
    valid_start_idx_full: int
    valid_end_idx_full: int
    full_len: int


@dataclass(frozen=True)
class _CachedFeatures:
    store: FeatureStore
    tokens_full: np.ndarray  # [T,N,D]
    z0_full: np.ndarray  # [T,D]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _cache_key(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_feature_arrays(
    token_npz: Path,
    *,
    feature_cache: Dict[Path, _CachedFeatures],
) -> _CachedFeatures:
    key = _cache_key(Path(token_npz))
    cached = feature_cache.get(key)
    if cached is not None:
        return cached

    store = open_feature_store(key)
    cached = _CachedFeatures(
        store=store,
        tokens_full=store.tokens.astype(np.float16, copy=False),
        z0_full=store.embeddings.astype(np.float32, copy=False),
    )
    feature_cache[key] = cached
    LOGGER.info(
        "Probe phase1 feature store=%s mode=%s tokens_shape=%s embeddings_shape=%s",
        key,
        store.mode,
        tuple(int(v) for v in store.tokens.shape),
        tuple(int(v) for v in store.embeddings.shape),
    )
    return cached


def _move_token_chunk_to_device(
    token_chunk: np.ndarray,
    *,
    device: torch.device,
    use_autocast: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(token_chunk).to(device=device)
    if not use_autocast and tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


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

    if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items() if isinstance(k, str)}
    if not any(isinstance(k, str) and k.startswith("pooler.") for k in state.keys()):
        state = {
            (k if (isinstance(k, str) and (k.startswith("pooler.") or k.startswith("linear."))) else f"pooler.{k}"): v
            for k, v in state.items()
        }
    state = {k: v for k, v in state.items() if isinstance(k, str) and not k.startswith("linear.")}
    if not state:
        raise RuntimeError(f"No pooler tensors found in {pooler_path}")
    return state


def _sha_from_state_dict(state: Dict[str, torch.Tensor], *, salt: str = "") -> str:
    buf = io.BytesIO()
    torch.save({"state": state}, buf)
    digest = hashlib.sha256()
    digest.update(buf.getvalue())
    if salt:
        digest.update(salt.encode("utf-8"))
    return digest.hexdigest()


def _prepare_stream(
    *,
    name: str,
    token_npz: Path,
    labels_path: Path,
    ignore_radius: int,
    smooth_sigma: float,
    boundary_index_mode: str,
    feature_cache: Dict[Path, _CachedFeatures],
) -> _StageStream:
    target_cfg = type(
        "_Cfg",
        (),
        {
            "ignore_radius": int(ignore_radius),
            "smooth_sigma": float(smooth_sigma),
            "boundary_index_mode": str(boundary_index_mode),
        },
    )()
    sup = _prepare_supervised_stream(
        name=str(name),
        embeddings_path=Path(token_npz),
        labels_path=Path(labels_path),
        cfg=target_cfg,
    )
    cached = _load_feature_arrays(Path(token_npz), feature_cache=feature_cache)
    s = int(sup.valid_start_idx_full)
    e = int(sup.valid_end_idx_full)
    tokens = cached.tokens_full[s : e + 1]
    z0 = cached.z0_full[s : e + 1]
    if int(tokens.shape[0]) != int(sup.embeddings.shape[0]):
        raise RuntimeError(
            f"Token/target length mismatch stream={name} tokens={tokens.shape[0]} targets={sup.embeddings.shape[0]}"
        )
    return _StageStream(
        name=str(name),
        tokens=tokens,
        z0=z0,
        y_start=sup.y_start.astype(np.float32, copy=False),
        y_end=sup.y_end.astype(np.float32, copy=False),
        y_boundary=sup.y_boundary.astype(np.float32, copy=False),
        y_cycle=sup.y_cycle.astype(np.float32, copy=False),
        y_phase=sup.y_phase.astype(np.float32, copy=False),
        y_class_label_id=sup.y_class_label_id.astype(np.int64, copy=False),
        mask_start_end=sup.mask_start_end.astype(np.float32, copy=False),
        mask_cycle=sup.mask_cycle.astype(np.float32, copy=False),
        mask_phase=sup.mask_phase.astype(np.float32, copy=False),
        mask_class=sup.mask_class.astype(np.float32, copy=False),
        mapped_cycles_idx=list(sup.mapped_cycles_idx),
        boundary_indices=list(sup.boundary_indices),
        supervised_mode=str(sup.supervised_mode),
        valid_start_idx_full=int(sup.valid_start_idx_full),
        valid_end_idx_full=int(sup.valid_end_idx_full),
        full_len=int(sup.full_len),
    )


def _local_cycles_in_chunk(
    mapped_cycles_idx: Sequence[Tuple[int, int]],
    *,
    start: int,
    end: int,
) -> List[Any]:
    return _visible_cycles_in_chunk(mapped_cycles_idx, start=int(start), end=int(end))


def _smooth_loss(z: torch.Tensor, mask_cycle: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    diff = (z[1:] - z[:-1]).pow(2).mean(dim=-1)
    pair_mask = mask_cycle[1:] * mask_cycle[:-1]
    return masked_mean(diff, pair_mask)


def _distill_loss(z: torch.Tensor, z0: torch.Tensor, mask_cycle: torch.Tensor) -> torch.Tensor:
    z_n = F.normalize(z, p=2, dim=-1, eps=1e-6)
    z0_n = F.normalize(z0, p=2, dim=-1, eps=1e-6)
    err = (z_n - z0_n).pow(2).mean(dim=-1)
    return masked_mean(err, mask_cycle)


def _set_probe_trainable(probe: nn.Module, mode: str) -> None:
    mode_n = str(mode or "all").strip().lower()
    for p in probe.parameters():
        p.requires_grad = False
    if mode_n == "all":
        for p in probe.parameters():
            p.requires_grad = True
        return

    for name, p in probe.named_parameters():
        lname = str(name).lower()
        if "query_tokens" in lname:
            p.requires_grad = True
        if ".norm" in lname or lname.startswith("norm") or "layernorm" in lname:
            p.requires_grad = True
        if mode_n in {"last_block", "last_blocks"} and (
            lname.startswith("cross_attention_block.") or lname.startswith("blocks.")
        ):
            p.requires_grad = True


def _set_tcn_trainable(tcn: nn.Module, *, mode: str, last_blocks: int) -> None:
    mode_n = str(mode or "frozen").strip().lower()
    for p in tcn.parameters():
        p.requires_grad = False
    if mode_n == "frozen":
        return
    if mode_n == "full":
        for p in tcn.parameters():
            p.requires_grad = True
        return
    if mode_n != "last_blocks":
        raise ValueError(f"Unsupported tcn_tune_mode={mode!r}")

    if hasattr(tcn, "out_proj") and isinstance(getattr(tcn, "out_proj"), nn.Module):
        for p in getattr(tcn, "out_proj").parameters():
            p.requires_grad = True
    if hasattr(tcn, "base_refines") and isinstance(getattr(tcn, "base_refines"), nn.ModuleList):
        for module in getattr(tcn, "base_refines"):
            for p in module.parameters():
                p.requires_grad = True
    if hasattr(tcn, "base_out") and isinstance(getattr(tcn, "base_out"), nn.ModuleList):
        for module in getattr(tcn, "base_out"):
            for p in module.parameters():
                p.requires_grad = True

    block_container = None
    if hasattr(tcn, "shared_blocks") and isinstance(getattr(tcn, "shared_blocks"), nn.Sequential):
        block_container = getattr(tcn, "shared_blocks")
    elif hasattr(tcn, "blocks") and isinstance(getattr(tcn, "blocks"), nn.Sequential):
        block_container = getattr(tcn, "blocks")
    if block_container is None:
        return
    n_blocks = int(max(1, last_blocks))
    for block in list(block_container.children())[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def _build_stream_sampling_weights(
    streams: Sequence[_StageStream],
    *,
    mode: str,
    power: float,
    min_weight: float,
) -> np.ndarray:
    if not streams:
        return np.zeros((0,), dtype=np.float64)
    mode_n = str(mode or "uniform").strip().lower()
    if mode_n == "uniform":
        return np.ones((len(streams),), dtype=np.float64)
    if mode_n != "sqrt_cycles":
        raise ValueError(f"Unsupported stream_sampling_mode={mode!r}")
    weights: List[float] = []
    for stream in streams:
        cycle_count = max(1, len(stream.mapped_cycles_idx))
        weight = max(float(min_weight), float(cycle_count) ** float(power))
        weights.append(float(weight))
    return np.asarray(weights, dtype=np.float64)


def _build_step_queue(
    streams: Sequence[_StageStream],
    *,
    chunks_per_stream: int,
    rng: np.random.RandomState,
    mode: str,
    power: float,
    min_weight: float,
) -> List[int]:
    num_streams = len(streams)
    if num_streams <= 0:
        return []
    total_steps = num_streams * max(1, int(chunks_per_stream))
    queue = [idx for idx in range(num_streams)]
    extra_steps = max(0, int(total_steps) - len(queue))
    if extra_steps > 0:
        weights = _build_stream_sampling_weights(
            streams,
            mode=str(mode),
            power=float(power),
            min_weight=float(min_weight),
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            weights = np.ones((num_streams,), dtype=np.float64)
            weight_sum = float(weights.sum())
        sampled = rng.choice(num_streams, size=int(extra_steps), replace=True, p=weights / weight_sum)
        queue.extend(int(idx) for idx in sampled.tolist())
    rng.shuffle(queue)
    return queue


def _build_optimizer(
    *,
    probe: nn.Module,
    tcn: nn.Module,
    config: ProbePhase1Config,
) -> tuple[torch.optim.Optimizer, str]:
    param_groups: List[Dict[str, object]] = []
    probe_params = list(probe.parameters())
    if probe_params:
        param_groups.append(
            {
                "params": probe_params,
                "lr": float(config.probe_lr),
                "weight_decay": float(config.weight_decay),
                "name": "probe",
            }
        )
    tcn_params = list(tcn.parameters()) if str(config.tcn_tune_mode or "frozen").strip().lower() != "frozen" else []
    if tcn_params:
        param_groups.append(
            {
                "params": tcn_params,
                "lr": float(config.tcn_lr),
                "weight_decay": float(config.tcn_weight_decay),
                "name": "tcn",
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for probe phase1.")

    optimizer_impl = "adamw"
    optimizer_kwargs: Dict[str, object] = {}
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
        optimizer_impl = "adamw_fused"
    try:
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    except (RuntimeError, TypeError) as exc:
        if optimizer_kwargs.get("fused"):
            LOGGER.info("Probe phase1 falling back to standard AdamW (fused unavailable: %s)", exc)
            optimizer = torch.optim.AdamW(param_groups)
            optimizer_impl = "adamw"
        else:
            raise
    return optimizer, optimizer_impl


def _count_trainable(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters() if p.requires_grad))


def _build_probe(input_embed_dim: int, device: torch.device, pooler_path: Path) -> nn.Module:
    from third_party.vjepa2_testing.src.pipeline.model_utils import build_classifier

    classifier = build_classifier(int(input_embed_dim), device)
    pooler_state = _load_pooler_state_for_classifier(pooler_path)
    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    if missing or unexpected:
        LOGGER.info("Probe load warnings missing=%s unexpected=%s", missing[:8], unexpected[:8])
    return classifier.pooler.to(device)


def _resolve_phase_head_indices(heads: Sequence[str]) -> Optional[Tuple[int, int]]:
    heads_n = [str(h).strip().lower() for h in heads]
    phase_sin_idx = next((idx for idx, name in enumerate(heads_n) if name == "phase_sin"), None)
    phase_cos_idx = next((idx for idx, name in enumerate(heads_n) if name == "phase_cos"), None)
    if phase_sin_idx is None or phase_cos_idx is None:
        return None
    return int(phase_sin_idx), int(phase_cos_idx)


def _normalize_temporal_structure_mode(value: Any) -> str:
    token = str(value or "cyclic").strip().lower().replace("-", "_")
    if token in {"cyclic", "event"}:
        return token
    return "cyclic"


def _resolve_class_head_info(
    heads: Sequence[str],
    multiclass_head: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if isinstance(multiclass_head, dict):
        enabled = multiclass_head.get("enabled")
        if enabled is not None and not bool(enabled):
            return None
    heads_n = [str(h).strip().lower() for h in heads]
    class_indices = [idx for idx, name in enumerate(heads_n) if name.startswith("class_")]
    if not class_indices:
        return None
    class_start = int(class_indices[0])
    class_stop = int(class_indices[-1] + 1)
    if class_indices != list(range(class_start, class_stop)):
        raise RuntimeError(f"Non-contiguous class heads in checkpoint: {class_indices}")

    raw_label_ids = multiclass_head.get("class_index_to_label_id") if isinstance(multiclass_head, dict) else None
    if not isinstance(raw_label_ids, list) or len(raw_label_ids) != (class_stop - class_start):
        return None

    label_id_to_class_index: Dict[int, int] = {}
    for class_index, raw_label_id in enumerate(raw_label_ids):
        try:
            label_id = int(raw_label_id)
        except (TypeError, ValueError):
            continue
        label_id_to_class_index[int(label_id)] = int(class_index)

    raw_class_weights = multiclass_head.get("class_weights") if isinstance(multiclass_head, dict) else None
    class_weights: Optional[np.ndarray] = None
    if isinstance(raw_class_weights, list) and len(raw_class_weights) == (class_stop - class_start):
        try:
            class_weights = np.asarray(raw_class_weights, dtype=np.float32)
        except (TypeError, ValueError):
            class_weights = None

    return {
        "class_start": int(class_start),
        "class_stop": int(class_stop),
        "num_classes": int(class_stop - class_start),
        "label_id_to_class_index": label_id_to_class_index,
        "class_weights": class_weights,
    }


def run_probe_phase1_finetune(
    *,
    token_npz_paths: Sequence[Path],
    labels_paths: Sequence[Path],
    stream_names: Sequence[str],
    tcn_checkpoint: Path,
    pooler_path: Path,
    encoder_model: str,
    encoder_checkpoint: str,
    output_dir: Path,
    config: ProbePhase1Config,
) -> ProbePhase1Result:
    if len(token_npz_paths) != len(labels_paths):
        raise ValueError("token_npz_paths and labels_paths length mismatch")
    if stream_names and len(stream_names) != len(token_npz_paths):
        raise ValueError("stream_names length mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(config.seed))

    tcn_payload = torch.load(tcn_checkpoint, map_location="cpu")
    if not isinstance(tcn_payload, dict):
        raise RuntimeError(f"Invalid TCN checkpoint payload: {tcn_checkpoint}")

    heads = tcn_payload.get("heads") if isinstance(tcn_payload.get("heads"), (list, tuple)) else []
    heads_n = [str(h).strip().lower() for h in heads]
    combine = "boundary" in heads_n and "start" not in heads_n and "end" not in heads_n
    phase_head_indices = _resolve_phase_head_indices(heads)
    use_phase_head = phase_head_indices is not None
    train_cfg_ckpt = tcn_payload.get("train_cfg", {}) if isinstance(tcn_payload.get("train_cfg"), dict) else {}
    temporal_mode = _normalize_temporal_structure_mode(
        config.temporal_structure_mode or train_cfg_ckpt.get("temporal_structure_mode", "cyclic")
    )
    event_mode = temporal_mode == "event"
    multiclass_head = tcn_payload.get("multiclass_head") if isinstance(tcn_payload.get("multiclass_head"), dict) else {}
    class_head_enabled = bool(multiclass_head.get("enabled", True)) if isinstance(multiclass_head, dict) else True
    class_head_info = _resolve_class_head_info(heads, multiclass_head)

    boundary_loss_name = str(train_cfg_ckpt.get("boundary_loss", "focal")).lower()
    gamma = float(train_cfg_ckpt.get("gamma", 2.0))
    pos_weight_start_end = float(train_cfg_ckpt.get("pos_weight_start_end", 10.0))
    phase_loss_name = str(train_cfg_ckpt.get("phase_loss", "mse")).lower()
    phase_huber_delta = float(train_cfg_ckpt.get("phase_huber_delta", 0.25))
    phase_loss_weight = 0.0 if event_mode else float(train_cfg_ckpt.get("phase_loss_weight", 0.15))
    ranking_loss_weight = 0.0 if event_mode else float(train_cfg_ckpt.get("ranking_loss_weight", 0.0))
    ranking_margin = float(train_cfg_ckpt.get("ranking_margin", 0.5))
    ranking_window_radius = int(train_cfg_ckpt.get("ranking_window_radius", 1))
    ignore_radius = int(train_cfg_ckpt.get("ignore_radius", 1))
    smooth_sigma = float(train_cfg_ckpt.get("smooth_sigma", 0.0))
    boundary_index_mode = str(config.boundary_index_mode or train_cfg_ckpt.get("boundary_index_mode", "nearest"))
    cyclece_weight = 0.0 if event_mode else float(config.cyclece_weight)
    class_weight = float(max(0.0, config.class_weight))
    use_cycle_loss = not event_mode
    class_weights_np = class_head_info.get("class_weights") if class_head_info is not None else None

    feature_cache: Dict[Path, _CachedFeatures] = {}
    streams: List[_StageStream] = []
    for idx, (token_npz, labels_path) in enumerate(zip(token_npz_paths, labels_paths)):
        name = stream_names[idx] if stream_names else f"stream_{idx:03d}"
        stream = _prepare_stream(
            name=str(name),
            token_npz=Path(token_npz),
            labels_path=Path(labels_path),
            ignore_radius=int(ignore_radius),
            smooth_sigma=float(smooth_sigma),
            boundary_index_mode=str(boundary_index_mode),
            feature_cache=feature_cache,
        )
        streams.append(stream)
        LOGGER.info(
            "Probe phase1 stream=%s T=%d mode=%s span=[%d,%d]",
            stream.name,
            int(stream.tokens.shape[0]),
            stream.supervised_mode,
            int(stream.valid_start_idx_full),
            int(stream.valid_end_idx_full),
        )
    if not streams:
        raise RuntimeError("No streams prepared for probe phase1 training")

    y_cycle_all = np.concatenate([s.y_cycle[s.mask_cycle > 0.5] for s in streams], axis=0)
    pos_weight_cycle = float(
        _resolve_cycle_pos_weight(
            y_cycle_all,
            type(
                "_Cfg",
                (),
                {
                    "pos_weight_cycle": None,
                    "cycle_weight_trigger_low": 0.35,
                    "cycle_weight_trigger_high": 0.65,
                    "cycle_weight_min": 1.0,
                    "cycle_weight_max": 5.0,
                },
            )(),
        )
    )

    input_embed_dim = int(streams[0].tokens.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not class_head_enabled:
        LOGGER.info("Probe phase1 checkpoint multiclass head is disabled; class CE is unavailable.")
    if class_weight > 0.0 and not class_head_enabled:
        raise RuntimeError(
            "Probe phase1 class CE requested (stage1_class_weight > 0) but checkpoint multiclass_head.enabled is false."
        )
    if class_weight > 0.0 and class_head_info is None:
        raise RuntimeError(
            "Probe phase1 class CE requested (stage1_class_weight > 0) but valid checkpoint class-head metadata was not found."
        )
    class_weights_t: Optional[torch.Tensor] = None
    if class_weights_np is not None:
        class_weights_t = torch.from_numpy(class_weights_np).to(device)
    probe = _build_probe(int(input_embed_dim), device, Path(pooler_path))
    prior_probe = copy.deepcopy(probe).to(device)
    prior_probe.eval()
    for p in prior_probe.parameters():
        p.requires_grad = False

    tcn = load_boundary_checkpoint(
        Path(tcn_checkpoint),
        input_dim=int(input_embed_dim),
        device=str(device),
        expected_seq_model=str(tcn_payload.get("seq_model", "tcn")),
    ).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _set_probe_trainable(probe, "light")
    _set_tcn_trainable(
        tcn,
        mode=str(config.tcn_tune_mode),
        last_blocks=int(config.tcn_last_blocks),
    )
    optimizer, optimizer_impl = _build_optimizer(probe=probe, tcn=tcn, config=config)
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16
    pos_w_se_t = torch.tensor(float(pos_weight_start_end), device=device)
    pos_w_cycle_t = torch.tensor(float(pos_weight_cycle), device=device)

    model_cfg = tcn_payload.get("model_cfg", {}) if isinstance(tcn_payload, dict) else {}
    left_ctx_raw = model_cfg.get("left_context") if isinstance(model_cfg, dict) else None
    if left_ctx_raw is None:
        bidirectional = bool(model_cfg.get("bidirectional")) if isinstance(model_cfg, dict) else False
        if bidirectional:
            left_ctx = 0
        else:
            left_ctx = 0
            dilations = model_cfg.get("dilations") if isinstance(model_cfg, dict) else None
            kernel_size = int(model_cfg.get("kernel_size", 3)) if isinstance(model_cfg, dict) else 3
            if isinstance(dilations, (list, tuple)) and dilations:
                left_ctx = (int(kernel_size) - 1) * 2 * int(sum(int(d) for d in dilations))
    else:
        left_ctx = int(left_ctx_raw)
    left_ctx = int(max(0, left_ctx))

    rng = np.random.RandomState(int(config.seed))
    best_loss = float("inf")
    best_probe_state: Optional[Dict[str, torch.Tensor]] = None
    best_tcn_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    history: List[Dict[str, float]] = []
    stream_sampling_weights = _build_stream_sampling_weights(
        streams,
        mode=str(config.stream_sampling_mode),
        power=float(config.stream_sampling_power),
        min_weight=float(config.stream_sampling_min_weight),
    )

    LOGGER.info(
        "Probe phase1 epochs=%d trainable_probe=%d trainable_tcn=%d left_ctx=%d temporal_mode=%s class_head=%s cycle_enabled=%s phase_enabled=%s rank_enabled=%s cyclece_enabled=%s tcn_tune_mode=%s tcn_last_blocks=%d optimizer=%s stream_sampling=%s[min=%.3f mean=%.3f max=%.3f]",
        int(config.epochs),
        _count_trainable(probe),
        _count_trainable(tcn),
        int(left_ctx),
        str(temporal_mode),
        bool(class_head_info is not None),
        bool(use_cycle_loss),
        bool(use_phase_head) and float(phase_loss_weight) > 0.0,
        float(ranking_loss_weight) > 0.0,
        float(cyclece_weight) > 0.0,
        str(config.tcn_tune_mode),
        int(config.tcn_last_blocks),
        optimizer_impl,
        str(config.stream_sampling_mode),
        float(stream_sampling_weights.min()) if stream_sampling_weights.size else 0.0,
        float(stream_sampling_weights.mean()) if stream_sampling_weights.size else 0.0,
        float(stream_sampling_weights.max()) if stream_sampling_weights.size else 0.0,
    )
    if bool(use_phase_head) and phase_head_indices is not None:
        LOGGER.info(
            "Probe phase1 phase-head indices phase_sin=%d phase_cos=%d",
            int(phase_head_indices[0]),
            int(phase_head_indices[1]),
        )

    for epoch in range(int(config.epochs)):
        if int(epoch) < int(config.stage1_last_block_epoch):
            probe_mode = "light"
        elif int(epoch) < int(config.stage1_all_epoch):
            probe_mode = "last_blocks"
        else:
            probe_mode = "all"
        _set_probe_trainable(probe, probe_mode)
        _set_tcn_trainable(
            tcn,
            mode=str(config.tcn_tune_mode),
            last_blocks=int(config.tcn_last_blocks),
        )
        LOGGER.info(
            "Probe phase1 epoch=%d probe_mode=%s trainable_probe=%d trainable_tcn=%d",
            epoch,
            probe_mode,
            _count_trainable(probe),
            _count_trainable(tcn),
        )

        probe.train()
        if any(bool(p.requires_grad) for p in tcn.parameters()):
            tcn.train()
        else:
            tcn.eval()

        step_queue = _build_step_queue(
            streams,
            chunks_per_stream=int(config.chunks_per_stream),
            rng=rng,
            mode=str(config.stream_sampling_mode),
            power=float(config.stream_sampling_power),
            min_weight=float(config.stream_sampling_min_weight),
        )

        totals = {
            "loss": 0.0,
            "loss_boundary": 0.0,
            "loss_start": 0.0,
            "loss_end": 0.0,
            "loss_cycle": 0.0,
            "loss_class": 0.0,
            "loss_phase": 0.0,
            "loss_rank": 0.0,
            "loss_cyclece": 0.0,
            "loss_smooth": 0.0,
            "loss_distill": 0.0,
        }
        steps = 0

        while step_queue:
            stream_idx = int(step_queue.pop())
            s = streams[stream_idx]
            T = int(s.tokens.shape[0])
            chunk_len = int(config.chunk_len if config.chunk_len > 0 else T)
            start = _select_chunk_start(
                rng=rng,
                T=T,
                chunk_len=chunk_len,
                boundary_indices=s.boundary_indices,
                neg_chunk_fraction=float(config.neg_chunk_fraction),
            )
            end = min(T, int(start + chunk_len))
            ctx_start = max(0, int(start - left_ctx))
            offset = int(start - ctx_start)
            chunk_len_eff = int(end - start)
            if chunk_len_eff <= 0:
                continue

            tokens_win = _move_token_chunk_to_device(
                s.tokens[ctx_start:end],
                device=device,
                use_autocast=bool(use_autocast),
            )
            y_cycle = torch.from_numpy(s.y_cycle[start:end]).to(device=device)
            mask_cycle = torch.from_numpy(s.mask_cycle[start:end]).to(device=device)
            mask_start_end = torch.from_numpy(s.mask_start_end[start:end]).to(device=device)
            raw_class_ids = torch.from_numpy(s.y_class_label_id[start:end]).to(device=device, dtype=torch.long)
            mask_class = torch.from_numpy(s.mask_class[start:end]).to(device=device)
            z0 = torch.from_numpy(s.z0[start:end]).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.cuda.amp.autocast(dtype=autocast_dtype) if use_autocast else contextlib.nullcontext()
            with autocast_ctx:
                z_win = probe(tokens_win).squeeze(1)
                logits_win = tcn(z_win.unsqueeze(0)).squeeze(0)
                z = z_win[offset : offset + chunk_len_eff]
                logits = logits_win[offset : offset + chunk_len_eff]

            if combine:
                logits_boundary = logits[:, 0]
                logits_cycle = logits[:, 1]
                y_boundary = torch.from_numpy(s.y_boundary[start:end]).to(device=device)
                loss_b_raw = _boundary_loss_with_logits(
                    logits=logits_boundary,
                    targets=y_boundary,
                    pos_weight=pos_w_se_t,
                    gamma=float(gamma),
                    loss_name=str(boundary_loss_name),
                )
                loss_boundary = masked_mean(loss_b_raw, mask_start_end)
                loss_start = loss_boundary.new_tensor(0.0)
                loss_end = loss_boundary.new_tensor(0.0)
            else:
                logits_start_end = logits[:, 0:2]
                logits_cycle = logits[:, 2]
                y_start = torch.from_numpy(s.y_start[start:end]).to(device=device)
                y_end = torch.from_numpy(s.y_end[start:end]).to(device=device)
                y_se = torch.stack([y_start, y_end], dim=-1)
                loss_se_raw = _boundary_loss_with_logits(
                    logits=logits_start_end,
                    targets=y_se,
                    pos_weight=pos_w_se_t,
                    gamma=float(gamma),
                    loss_name=str(boundary_loss_name),
                )
                loss_start = masked_mean(loss_se_raw[:, 0], mask_start_end)
                loss_end = masked_mean(loss_se_raw[:, 1], mask_start_end)
                loss_boundary = loss_start + loss_end

            if use_cycle_loss:
                loss_cycle_raw = bce_with_logits(
                    logits_cycle,
                    y_cycle,
                    pos_weight=pos_w_cycle_t if float(pos_weight_cycle) != 1.0 else None,
                    reduction="none",
                )
                loss_cycle = masked_mean(loss_cycle_raw, mask_cycle)
            else:
                loss_cycle = logits.new_tensor(0.0)

            loss_class = logits.new_tensor(0.0)
            if class_head_info is not None and class_weight > 0.0:
                class_targets = torch.zeros_like(raw_class_ids)
                label_id_to_class_index = class_head_info["label_id_to_class_index"]
                for label_id, class_index in label_id_to_class_index.items():
                    class_targets = torch.where(
                        raw_class_ids == int(label_id),
                        torch.full_like(raw_class_ids, int(class_index)),
                        class_targets,
                    )
                logits_class = logits[:, int(class_head_info["class_start"]) : int(class_head_info["class_stop"])]
                loss_class = _masked_multiclass_ce(
                    logits=logits_class,
                    targets=class_targets,
                    mask=mask_class,
                    class_weights=class_weights_t,
                )

            loss_phase = logits.new_tensor(0.0)
            if bool(use_phase_head) and float(phase_loss_weight) > 0.0 and phase_head_indices is not None:
                phase_sin_idx = int(phase_head_indices[0])
                phase_cos_idx = int(phase_head_indices[1])
                phase_max_idx = max(phase_sin_idx, phase_cos_idx)
                if int(logits.shape[-1]) <= int(phase_max_idx):
                    raise RuntimeError(
                        f"Phase head index out of bounds: logits_dim={int(logits.shape[-1])} "
                        f"phase_sin_idx={phase_sin_idx} phase_cos_idx={phase_cos_idx}"
                    )
                y_phase = torch.from_numpy(s.y_phase[start:end]).to(device=device)
                mask_phase = torch.from_numpy(s.mask_phase[start:end]).to(device=device)
                logits_phase = torch.stack([logits[:, phase_sin_idx], logits[:, phase_cos_idx]], dim=-1)
                loss_phase = _phase_loss_masked(
                    pred_phase_logits=logits_phase,
                    target_phase=y_phase,
                    mask_phase=mask_phase,
                    loss_name=str(phase_loss_name),
                    huber_delta=float(phase_huber_delta),
                )

            local_cycles = _local_cycles_in_chunk(s.mapped_cycles_idx, start=start, end=end)
            loss_rank = logits.new_tensor(0.0)
            if float(ranking_loss_weight) > 0.0 and local_cycles:
                base_channels = 2 if combine else 3
                loss_rank = _ranking_loss_start_end(
                    logits=logits[:, :base_channels],
                    mapped_cycles_idx=local_cycles,
                    combine=bool(combine),
                    margin=float(ranking_margin),
                    radius=int(ranking_window_radius),
                )

            loss_cyclece = logits.new_tensor(0.0)
            if float(cyclece_weight) > 0.0 and local_cycles:
                base_channels = 2 if combine else 3
                loss_cyclece = _cycle_ce_loss(
                    logits=logits[:, :base_channels],
                    mapped_cycles_idx=local_cycles,
                    combine=bool(combine),
                    tau=float(config.cyclece_tau),
                    radius=int(config.cyclece_radius),
                )

            loss_smooth = logits.new_tensor(0.0)
            if float(config.smooth_weight) > 0.0:
                loss_smooth = _smooth_loss(z, mask_cycle)

            loss_distill = logits.new_tensor(0.0)
            if float(config.distill_weight) > 0.0:
                loss_distill = _distill_loss(z, z0, mask_cycle)

            loss = (
                loss_boundary
                + (loss_class * float(class_weight))
                + (loss_phase * float(phase_loss_weight))
                + (loss_rank * float(ranking_loss_weight))
                + (loss_cyclece * float(cyclece_weight))
                + (loss_smooth * float(config.smooth_weight))
                + (loss_distill * float(config.distill_weight))
            )
            if use_cycle_loss:
                loss = loss + loss_cycle

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected epoch={epoch}")
            loss.backward()
            if float(config.grad_clip_norm) > 0:
                params = [p for p in list(probe.parameters()) + list(tcn.parameters()) if p.requires_grad]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["loss_boundary"] += float(loss_boundary.item())
            totals["loss_start"] += float(loss_start.item())
            totals["loss_end"] += float(loss_end.item())
            totals["loss_cycle"] += float(loss_cycle.item())
            totals["loss_class"] += float(loss_class.item())
            totals["loss_phase"] += float(loss_phase.item())
            totals["loss_rank"] += float(loss_rank.item())
            totals["loss_cyclece"] += float(loss_cyclece.item())
            totals["loss_smooth"] += float(loss_smooth.item())
            totals["loss_distill"] += float(loss_distill.item())
            steps += 1

        if steps <= 0:
            raise RuntimeError("No optimization steps executed in probe phase1")

        row = {"epoch": float(epoch)}
        for key, value in totals.items():
            row[key] = float(value / float(steps))
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = float(row["loss"])
            best_epoch = int(epoch)
            best_probe_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            best_tcn_state = {k: v.detach().cpu().clone() for k, v in tcn.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == int(config.epochs) - 1:
            LOGGER.info(
                "Probe phase1 epoch=%d loss=%.4f boundary=%.4f cycle=%.4f class=%.4f phase=%.4f rank=%.4f cyclece=%.4f smooth=%.4f distill=%.4f",
                epoch,
                row["loss"],
                row["loss_boundary"],
                row["loss_cycle"],
                row["loss_class"],
                row["loss_phase"],
                row["loss_rank"],
                row["loss_cyclece"],
                row["loss_smooth"],
                row["loss_distill"],
            )

    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)
    if best_tcn_state is not None:
        tcn.load_state_dict(best_tcn_state)
    if bool(config.fail_if_best_epoch_zero) and int(config.epochs) > 1 and int(best_epoch) <= 0:
        raise RuntimeError("Probe phase1 best_epoch=0 and fail_if_best_epoch_zero is enabled")

    probe_state = {f"pooler.{k}": v.detach().cpu() for k, v in probe.state_dict().items()}
    pooler_sha = _sha_from_state_dict(probe_state, salt=":probe_phase1")
    probe_ckpt = output_dir / "probe_pretrained.pt"
    torch.save(
        {
            "pooler_state": probe_state,
            "metadata": {
                "pooler_sha": str(pooler_sha),
                "encoder_model": str(encoder_model),
                "encoder_checkpoint": str(encoder_checkpoint),
                "source": "dense_temporal_probe_phase1",
                "stage": "probe_phase1",
            },
        },
        probe_ckpt,
    )

    tcn_payload_out = dict(tcn_payload)
    tcn_payload_out["model_state"] = {k: v.detach().cpu() for k, v in tcn.state_dict().items()}
    tcn_payload_out["probe_finetune"] = {
        "stage": "probe_phase1",
        "temporal_structure_mode": str(temporal_mode),
        "effective_losses": {
            "boundary": True,
            "class": bool(class_head_info is not None and class_weight > 0.0),
            "cycle": bool(use_cycle_loss),
            "phase": bool(use_phase_head) and float(phase_loss_weight) > 0.0,
            "ranking": float(ranking_loss_weight) > 0.0,
            "cyclece": float(cyclece_weight) > 0.0,
            "smooth": float(config.smooth_weight) > 0.0,
            "distill": float(config.distill_weight) > 0.0,
        },
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "cyclece_weight": float(cyclece_weight),
        "smooth_weight": float(config.smooth_weight),
        "distill_weight": float(config.distill_weight),
        "class_weight": float(class_weight),
    }
    tcn_ckpt = output_dir / "boundary_model.pt"
    torch.save(tcn_payload_out, tcn_ckpt)

    metrics = output_dir / "train_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "stage": "probe_phase1",
                "temporal_structure_mode": str(temporal_mode),
                "effective_losses": tcn_payload_out["probe_finetune"]["effective_losses"],
                "best_epoch": int(best_epoch),
                "best_loss": float(best_loss),
                "history": history,
                "trainable_probe": int(_count_trainable(probe)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ProbePhase1Result(
        probe_checkpoint=probe_ckpt,
        tcn_checkpoint=tcn_ckpt,
        metrics_path=metrics,
        best_epoch=int(best_epoch),
        best_loss=float(best_loss),
    )
