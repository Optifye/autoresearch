from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train


def test_stage_budget_split_scales_total_budget(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TCN_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_PROBE_STAGE_SECONDS", raising=False)

    tcn_default, probe_default = train.resolve_stage_budget_seconds(600.0)
    assert tcn_default == 300.0
    assert probe_default == 300.0

    tcn_scaled, probe_scaled = train.resolve_stage_budget_seconds(10.0)
    assert tcn_scaled == 5.0
    assert probe_scaled == 5.0


def test_stage_budget_split_respects_requested_ratio(monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_TCN_STAGE_SECONDS", "400")
    monkeypatch.setenv("AUTORESEARCH_PROBE_STAGE_SECONDS", "200")

    tcn_seconds, probe_seconds = train.resolve_stage_budget_seconds(60.0)
    assert tcn_seconds == 40.0
    assert probe_seconds == 20.0


def test_is_better_metrics_prefers_proxy_then_false_then_legacy_then_count_then_timing() -> None:
    incumbent = {
        "val_pair_f1": 0.90,
        "val_proxy_macro_f1": 0.90,
        "val_proxy_total_false_count": 4.0,
        "val_legacy_pair_f1": 0.70,
        "val_count_mae": 0.20,
        "val_start_mae_ms": 110.0,
        "val_end_mae_ms": 150.0,
    }
    better_proxy = {
        "val_pair_f1": 0.91,
        "val_proxy_macro_f1": 0.91,
        "val_proxy_total_false_count": 10.0,
        "val_legacy_pair_f1": 0.60,
        "val_count_mae": 0.30,
        "val_start_mae_ms": 200.0,
        "val_end_mae_ms": 220.0,
    }
    better_false = {
        "val_pair_f1": 0.90,
        "val_proxy_macro_f1": 0.90,
        "val_proxy_total_false_count": 3.0,
        "val_legacy_pair_f1": 0.60,
        "val_count_mae": 0.30,
        "val_start_mae_ms": 200.0,
        "val_end_mae_ms": 220.0,
    }
    better_legacy = {
        "val_pair_f1": 0.90,
        "val_proxy_macro_f1": 0.90,
        "val_proxy_total_false_count": 4.0,
        "val_legacy_pair_f1": 0.71,
        "val_count_mae": 0.30,
        "val_start_mae_ms": 200.0,
        "val_end_mae_ms": 220.0,
    }
    better_count = {
        "val_pair_f1": 0.90,
        "val_proxy_macro_f1": 0.90,
        "val_proxy_total_false_count": 4.0,
        "val_legacy_pair_f1": 0.70,
        "val_count_mae": 0.10,
        "val_start_mae_ms": 180.0,
        "val_end_mae_ms": 200.0,
    }
    better_timing = {
        "val_pair_f1": 0.90,
        "val_proxy_macro_f1": 0.90,
        "val_proxy_total_false_count": 4.0,
        "val_legacy_pair_f1": 0.70,
        "val_count_mae": 0.20,
        "val_start_mae_ms": 90.0,
        "val_end_mae_ms": 120.0,
    }
    worse = {
        "val_pair_f1": 0.89,
        "val_proxy_macro_f1": 0.89,
        "val_proxy_total_false_count": 0.0,
        "val_legacy_pair_f1": 0.99,
        "val_count_mae": 0.01,
        "val_start_mae_ms": 10.0,
        "val_end_mae_ms": 20.0,
    }

    assert train._is_better_metrics(better_proxy, incumbent) is True
    assert train._is_better_metrics(better_false, incumbent) is True
    assert train._is_better_metrics(better_legacy, incumbent) is True
    assert train._is_better_metrics(better_count, incumbent) is True
    assert train._is_better_metrics(better_timing, incumbent) is True
    assert train._is_better_metrics(worse, incumbent) is False


def test_is_better_metrics_falls_back_to_legacy_pair_f1_then_count_then_timing() -> None:
    incumbent = {
        "val_pair_f1": 0.80,
        "val_count_mae": 0.20,
        "val_start_mae_ms": 110.0,
        "val_end_mae_ms": 150.0,
    }
    better_f1 = {
        "val_pair_f1": 0.81,
        "val_count_mae": 0.30,
        "val_start_mae_ms": 200.0,
        "val_end_mae_ms": 220.0,
    }
    worse_f1 = {
        "val_pair_f1": 0.79,
        "val_count_mae": 0.01,
        "val_start_mae_ms": 10.0,
        "val_end_mae_ms": 20.0,
    }

    assert train._is_better_metrics(better_f1, incumbent) is True
    assert train._is_better_metrics(worse_f1, incumbent) is False


def test_threshold_proxy_eval_promotes_macro_f1_and_counts_false_fires() -> None:
    record = train.SegmentRecord(
        segment_id="seg-1",
        split="val",
        video_id="vid-1",
        camera_id="cam-1",
        source_run_dir="/tmp/source",
        feature_path="/tmp/feature.npz",
        label_path="/tmp/label.json",
        pooler_checkpoint="/tmp/pooler.pt",
        pooler_sha="pooler",
        embedding_dim=4,
        token_dim=4,
        tokens_per_window=1,
        num_total_windows=5,
        fps=1.0,
        supervised_start_ms=0,
        supervised_end_ms=400,
        supervised_start_idx=0,
        supervised_end_idx=4,
        eval_start_ms=0,
        eval_end_ms=400,
        eval_start_idx=0,
        eval_end_idx=4,
        event_pairs_ms=((100, 300),),
    )
    logits = np.asarray(
        [
            [-4.0, -4.0, -4.0],
            [4.0, -4.0, -4.0],
            [4.0, -4.0, -4.0],
            [-4.0, 4.0, -4.0],
            [-4.0, -4.0, -4.0],
        ],
        dtype=np.float32,
    )
    segment = train.ValidationEvalSegment(
        record=record,
        timestamps_ms=np.asarray([0, 100, 200, 300, 400], dtype=np.int64),
        logits=logits,
        mapped_cycles_idx=((1, 3),),
    )

    metrics = train._evaluate_threshold_proxy_segments([segment], heads=("start", "end", "cycle"))

    assert metrics["val_proxy_macro_f1"] == 1.0
    assert metrics["val_proxy_start_f1"] == 1.0
    assert metrics["val_proxy_end_f1"] == 1.0
    assert metrics["val_proxy_total_false_count"] == 0.0
    assert metrics["val_proxy_start_duplicate_excess"] == 1.0


def test_resolve_left_context_respects_bidirectional_and_hybrid() -> None:
    assert (
        train._resolve_left_context_from_tcn_payload(
            {
                "seq_model": "hybrid_tcn_lstm",
                "model_cfg": {"kernel_size": 5, "dilations": [1, 2, 4], "bidirectional": False},
                "train_cfg": {},
            }
        )
        == 0
    )
    assert (
        train._resolve_left_context_from_tcn_payload(
            {
                "seq_model": "tcn",
                "model_cfg": {"kernel_size": 5, "dilations": [1, 2, 4], "bidirectional": True},
                "train_cfg": {},
            }
        )
        == 0
    )
    assert (
        train._resolve_left_context_from_tcn_payload(
            {
                "seq_model": "tcn",
                "model_cfg": {"kernel_size": 5, "dilations": [1, 2, 4], "bidirectional": False},
                "train_cfg": {},
            }
        )
        == (5 - 1) * 2 * (1 + 2 + 4)
    )


def test_resolve_source_run_dir_prefers_manifest_source(tmp_path, monkeypatch) -> None:
    fresh_source = tmp_path / "fresh" / "dense_temporal"
    for child in ("features", "labels"):
        (fresh_source / child).mkdir(parents=True, exist_ok=True)
    (fresh_source / "snapshot.json").write_text("{}", encoding="utf-8")
    (fresh_source / "resolved_config.json").write_text("{}", encoding="utf-8")

    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(
        '{"source_run_dirs": ["' + str(fresh_source) + '"]}',
        encoding="utf-8",
    )
    monkeypatch.delenv("AUTORESEARCH_MINDA_SUBASSEMBLY_SOURCE_RUN_DIR", raising=False)

    resolved = train._resolve_source_run_dir(cache_root)
    assert resolved == fresh_source.resolve()
