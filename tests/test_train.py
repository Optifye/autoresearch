from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train


def _record(*, video_id: str, camera_id: str, segment_id: str) -> train.SegmentRecord:
    return train.SegmentRecord(
        segment_id=segment_id,
        split="train",
        video_id=video_id,
        camera_id=camera_id,
        source_run_dir="/tmp/source",
        feature_path="/tmp/features.npz",
        label_path="/tmp/labels.json",
        pooler_checkpoint="/tmp/pooler.pt",
        pooler_sha="sha",
        embedding_dim=1024,
        token_dim=1024,
        tokens_per_window=64,
        num_total_windows=128,
        fps=25.0,
        supervised_start_ms=0,
        supervised_end_ms=1000,
        supervised_start_idx=0,
        supervised_end_idx=10,
        eval_start_ms=0,
        eval_end_ms=1000,
        eval_start_idx=0,
        eval_end_idx=10,
        event_pairs_ms=((0, 500),),
    )


def test_stage_budget_split_scales_total_budget(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TCN_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_PROBE_STAGE_SECONDS", raising=False)

    tcn_default, probe_default = train.resolve_stage_budget_seconds(600.0)
    assert tcn_default == 300.0
    assert probe_default == 300.0

    tcn_scaled, probe_scaled = train.resolve_stage_budget_seconds(10.0)
    assert tcn_scaled == 5.0
    assert probe_scaled == 5.0


def test_camera_balanced_split_handles_singleton_cameras() -> None:
    records = [
        _record(video_id="v1", camera_id="cam_a", segment_id="v1__seg000"),
        _record(video_id="v2", camera_id="cam_a", segment_id="v2__seg000"),
        _record(video_id="v3", camera_id="cam_b", segment_id="v3__seg000"),
        _record(video_id="v4", camera_id="cam_b", segment_id="v4__seg000"),
        _record(video_id="v5", camera_id="cam_c", segment_id="v5__seg000"),
    ]

    split = train.build_camera_balanced_split(records, seed=42)

    assert len(split.train_videos) + len(split.val_videos) == 5
    assert len(split.val_videos) == 3
    assert split.camera_total_counts == {"cam_a": 2, "cam_b": 2, "cam_c": 1}
    assert split.camera_val_counts["cam_a"] == 1
    assert split.camera_val_counts["cam_b"] == 1
    assert split.camera_val_counts["cam_c"] in {0, 1}
    assert len(split.val_eval_records) == len(split.val_records)


def test_bundle_metrics_uses_worst_space_as_primary() -> None:
    bundle = train._bundle_metrics(
        {
            "minda-button-tcn": {
                "val_pair_f1": 0.9,
                "val_count_mae": 0.2,
                "val_start_mae_ms": 100.0,
                "val_end_mae_ms": 120.0,
            },
            "minda-subassembly-tcn": {
                "val_pair_f1": 0.8,
                "val_count_mae": 0.4,
                "val_start_mae_ms": 150.0,
                "val_end_mae_ms": 170.0,
            },
        }
    )

    assert bundle.primary_metric == 0.8
    assert round(bundle.mean_pair_f1, 6) == 0.85
    assert round(bundle.mean_count_mae, 6) == 0.3


def test_should_replace_eval_prefers_worse_case_improvement() -> None:
    best = train._bundle_metrics(
        {
            "minda-button-tcn": {
                "val_pair_f1": 0.9,
                "val_count_mae": 0.2,
                "val_start_mae_ms": 100.0,
                "val_end_mae_ms": 120.0,
            },
            "minda-subassembly-tcn": {
                "val_pair_f1": 0.8,
                "val_count_mae": 0.2,
                "val_start_mae_ms": 100.0,
                "val_end_mae_ms": 120.0,
            },
        }
    )
    better = train._bundle_metrics(
        {
            "minda-button-tcn": {
                "val_pair_f1": 0.91,
                "val_count_mae": 0.3,
                "val_start_mae_ms": 100.0,
                "val_end_mae_ms": 120.0,
            },
            "minda-subassembly-tcn": {
                "val_pair_f1": 0.82,
                "val_count_mae": 0.3,
                "val_start_mae_ms": 100.0,
                "val_end_mae_ms": 120.0,
            },
        }
    )
    worse = train._bundle_metrics(
        {
            "minda-button-tcn": {
                "val_pair_f1": 0.95,
                "val_count_mae": 0.1,
                "val_start_mae_ms": 90.0,
                "val_end_mae_ms": 100.0,
            },
            "minda-subassembly-tcn": {
                "val_pair_f1": 0.79,
                "val_count_mae": 0.1,
                "val_start_mae_ms": 90.0,
                "val_end_mae_ms": 100.0,
            },
        }
    )

    assert train._should_replace_eval(best, better) is True
    assert train._should_replace_eval(best, worse) is False


def test_resolve_left_context_falls_back_from_kernel_and_dilations() -> None:
    left_ctx = train._resolve_left_context_from_tcn_payload(
        {
            "seq_model": "tcn",
            "model_cfg": {
                "kernel_size": 5,
                "dilations": [1, 2, 4],
                "bidirectional": False,
            },
            "train_cfg": {},
        }
    )
    assert left_ctx == (5 - 1) * 2 * (1 + 2 + 4)
