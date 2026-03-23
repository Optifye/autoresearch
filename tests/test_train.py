from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train


def test_dense_temporal_model_pooled_mode() -> None:
    model = train.DenseTemporalModel(
        input_dim=3,
        pooler_checkpoint=None,
        representation_mode="pooled_z0",
        pooler_tune_mode="off",
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 5, 3)
    logits, features = model(x)
    assert logits.shape == (2, 5, 3)
    assert features.shape == (2, 5, 3)
    assert train.count_trainable_params(model) > 0


def test_lr_schedule_endpoints() -> None:
    start = train.get_lr_multiplier(0.0)
    end = train.get_lr_multiplier(1.0)
    assert start == 0.0
    assert round(end, 6) == round(float(train.FINAL_LR_FRAC), 6)


def test_stage_budget_split_scales_total_budget(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TCN_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_POOLER_STAGE_SECONDS", raising=False)
    tcn_default, pooler_default = train.resolve_stage_budget_seconds(600.0)
    assert tcn_default == 180.0
    assert pooler_default == 420.0

    tcn_scaled, pooler_scaled = train.resolve_stage_budget_seconds(10.0)
    assert round(tcn_scaled, 6) == 3.0
    assert round(pooler_scaled, 6) == 7.0


def test_event_multiclass_targets_and_metrics(tmp_path, monkeypatch) -> None:
    feature_path = tmp_path / "video__features.npz"
    np.savez(
        feature_path,
        tokens=np.arange(5 * 2 * 3, dtype=np.float16).reshape(5, 2, 3),
        embeddings=np.arange(15, dtype=np.float32).reshape(5, 3),
        timestamps_ms=np.arange(5, dtype=np.int64) * 1000,
        embedding_kind=np.array("base"),
        model_name=np.array("ssv2"),
        clip_len=np.array(16),
        stride=np.array(10),
        frame_skip=np.array(0),
        timestamp_alignment=np.array("center"),
        pooler_sha=np.array("pooler_sha"),
        camera_id=np.array("camera"),
        video_id=np.array("video"),
        fps=np.array(1.0),
    )
    label_path = tmp_path / "video__seg000.json"
    label_path.write_text(
        json.dumps(
            {
                "video_id": "video",
                "camera_id": "camera",
                "fps": 1.0,
                "label_map": {
                    "idle": 0,
                    "machine_idle": 1,
                    "machine_working": 4,
                    "machine_empty": 5,
                    "ignore": 6,
                },
                "action_labels": [
                    {"label_name": "machine_idle", "label_id": 1},
                    {"label_name": "machine_working", "label_id": 4},
                    {"label_name": "machine_empty", "label_id": 5},
                ],
                "class_targets_rle": [
                    [1, 0, 1],
                    [4, 2, 3],
                    [5, 4, 4],
                ],
            }
        ),
        encoding="utf-8",
    )

    record = train.SegmentRecord(
        segment_id="video__seg000",
        split="train",
        video_id="video",
        camera_id="camera",
        source_run_dir=str(tmp_path),
        feature_path=str(feature_path),
        label_path=str(label_path),
        pooler_checkpoint="",
        pooler_sha="sha",
        embedding_dim=3,
        token_dim=3,
        tokens_per_window=2,
        num_total_windows=5,
        fps=1.0,
        supervised_start_ms=0,
        supervised_end_ms=4000,
        supervised_start_idx=0,
        supervised_end_idx=4,
        eval_start_ms=None,
        eval_end_ms=None,
        eval_start_idx=None,
        eval_end_idx=None,
        event_pairs_ms=tuple(),
    )

    schema = train.resolve_event_class_schema([record])
    segment = train.build_supervised_segment(
        record,
        use_eval_span=False,
        preload_pooled=True,
        task_mode="event_multiclass",
        event_schema=schema,
    )

    assert schema.label_names == ("machine_idle", "machine_working", "machine_empty")
    assert segment.y_class is not None
    assert segment.mask_class is not None
    assert segment.y_class.tolist() == [0, 0, 1, 1, 2]
    assert segment.mask_class.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]

    monkeypatch.setattr(
        train,
        "_forward_segment_logits",
        lambda model, seg, device, eval_token_chunk: np.asarray(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        ),
    )
    metrics = train.evaluate_event_model(
        model=train.DenseTemporalModel(
            input_dim=3,
            pooler_checkpoint=None,
            representation_mode="pooled_z0",
            pooler_tune_mode="off",
            device=torch.device("cpu"),
            output_dim=3,
        ),
        val_segments=[segment],
        device=torch.device("cpu"),
        eval_token_chunk=8,
        event_schema=schema,
    )

    assert metrics["val_accuracy"] == 0.8
    assert metrics["val_recall_machine_idle"] == 0.5
    assert metrics["val_support_machine_empty"] == 1.0


def test_event_loss_ignores_masked_negative_targets() -> None:
    logits = torch.zeros((1, 3, 3), dtype=torch.float32)
    targets = {
        "y_class": torch.tensor([[0, -1, 2]], dtype=torch.int64),
        "mask_class": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        "valid_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    }
    loss, stats = train.compute_event_loss(logits, targets, class_weights=None)
    assert torch.isfinite(loss)
    assert stats["loss_total"] >= 0.0
