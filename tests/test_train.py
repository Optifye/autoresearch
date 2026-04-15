from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train
from autoresearch_vjepa.cache_contract import SegmentRecord


def _fake_record(tmp_path: Path) -> SegmentRecord:
    pooler_path = tmp_path / "pooler.pt"
    pooler_path.write_bytes(b"pooler")
    feature_path = tmp_path / "features.npz"
    feature_path.write_bytes(b"npz")
    label_path = tmp_path / "labels.json"
    label_path.write_text("{}", encoding="utf-8")
    return SegmentRecord(
        segment_id="seg-0",
        split="train",
        video_id="vid-0",
        camera_id="cam-0",
        source_run_dir=str(tmp_path / "source"),
        feature_path=str(feature_path),
        label_path=str(label_path),
        pooler_checkpoint=str(pooler_path),
        pooler_sha="pooler_sha",
        embedding_dim=1024,
        token_dim=1024,
        tokens_per_window=4,
        num_total_windows=16,
        fps=25.0,
        supervised_start_ms=0,
        supervised_end_ms=1000,
        supervised_start_idx=0,
        supervised_end_idx=10,
        eval_start_ms=0,
        eval_end_ms=1000,
        eval_start_idx=0,
        eval_end_idx=10,
        event_pairs_ms=((100, 700),),
    )


def _prod_checkpoint_payload(*, dilations: tuple[int, ...] = (1, 2, 4)) -> dict:
    return {
        "seq_model": "tcn",
        "heads": ["start", "end", "cycle"],
        "model_cfg": {
            "hidden_dim": 128,
            "kernel_size": 5,
            "dropout": 0.1,
            "use_layernorm": True,
            "dilations": list(dilations),
            "bidirectional": True,
            "task_specific_heads": True,
        },
        "train_cfg": {
            "boundary_loss": "focal",
            "gamma": 2.0,
            "pos_weight_start_end": 10.0,
            "pos_weight_cycle": 1.0,
            "ignore_radius": 1,
            "smooth_sigma": 0.0,
            "combine_start_end": False,
            "boundary_index_mode": "ordered_nearest",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 200,
            "chunk_len": 256,
            "chunks_per_epoch": 64,
            "neg_chunk_fraction": 0.40,
            "grad_clip_norm": 1.0,
            "stage1_enabled": True,
            "stage1_epochs": 20,
            "stage1_probe_lr": 2.6e-5,
            "stage1_last_block_epoch": 0,
            "stage1_all_epoch": 999,
            "stage1_chunks_per_stream": 12,
            "stage1_neg_chunk_fraction": 0.25,
            "stage1_tcn_tune_mode": "frozen",
            "stage1_tcn_last_blocks": 1,
            "stage1_tcn_lr": 5e-5,
            "stage1_tcn_weight_decay": 1e-4,
            "stage1_stream_sampling_mode": "uniform",
            "stage1_stream_sampling_power": 0.5,
            "stage1_stream_sampling_min_weight": 1.0,
            "stage1_cyclece_weight": 1.0,
            "stage1_smooth_weight": 0.05,
            "stage1_distill_weight": 0.10,
            "stage1_class_weight": 0.0,
            "stage1_fail_if_best_epoch_zero": False,
        },
    }


def _prod_summary_payload() -> dict:
    best_metrics = {
        "halo16_eval": {
            "metrics": {
                "val_pair_f1": 0.952,
                "val_count_mae": 0.40,
            },
            "timing_mae_ms": 125.0,
        },
        "legacy_prod_eval": {
            "metrics": {
                "val_pair_f1": 0.940,
            }
        },
        "chunk32_eval": {
            "metrics": {
                "val_pair_f1": 0.933,
            }
        },
        "threshold_proxy_eval": {
            "val_proxy_macro_f1": 0.901,
            "val_proxy_total_false_count": 17.0,
        },
    }
    return {
        "elapsed_seconds": 123.4,
        "baseline": {
            "halo16_eval": {
                "metrics": {
                    "val_pair_f1": 0.915,
                }
            }
        },
        "best_by_halo16": {
            "epoch": 4,
            "metrics": best_metrics,
        },
        "final_artifacts": {
            "probe_checkpoint": "/tmp/probe.pt",
            "tcn_checkpoint": "/tmp/boundary_model.pt",
        },
    }


def test_stage_budget_split_defaults_phase1_only(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TCN_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_PROBE_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_TIME_BUDGET_SECONDS", raising=False)

    tcn_default, probe_default = train.resolve_stage_budget_seconds(300.0)
    assert tcn_default == 0.0
    assert probe_default == 300.0

    monkeypatch.setenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "60")
    tcn_scaled, probe_scaled = train.resolve_stage_budget_seconds(60.0)
    assert tcn_scaled == 0.0
    assert probe_scaled == 60.0


def test_stage_budget_split_respects_requested_ratio(monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_TCN_STAGE_SECONDS", "30")
    monkeypatch.setenv("AUTORESEARCH_PROBE_STAGE_SECONDS", "270")
    monkeypatch.setenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "60")

    tcn_seconds, probe_seconds = train.resolve_stage_budget_seconds(60.0)
    assert tcn_seconds == 6.0
    assert probe_seconds == 54.0


def test_halo_selection_tuple_prefers_halo_then_false_then_count_then_timing() -> None:
    incumbent = {
        "halo16_eval": {"metrics": {"val_pair_f1": 0.90, "val_count_mae": 0.40}, "timing_mae_ms": 140.0},
        "threshold_proxy_eval": {"val_proxy_total_false_count": 20.0},
    }
    better_halo = {
        "halo16_eval": {"metrics": {"val_pair_f1": 0.91, "val_count_mae": 99.0}, "timing_mae_ms": 999.0},
        "threshold_proxy_eval": {"val_proxy_total_false_count": 999.0},
    }
    better_false = {
        "halo16_eval": {"metrics": {"val_pair_f1": 0.90, "val_count_mae": 99.0}, "timing_mae_ms": 999.0},
        "threshold_proxy_eval": {"val_proxy_total_false_count": 19.0},
    }
    better_count = {
        "halo16_eval": {"metrics": {"val_pair_f1": 0.90, "val_count_mae": 0.30}, "timing_mae_ms": 999.0},
        "threshold_proxy_eval": {"val_proxy_total_false_count": 20.0},
    }
    better_timing = {
        "halo16_eval": {"metrics": {"val_pair_f1": 0.90, "val_count_mae": 0.40}, "timing_mae_ms": 120.0},
        "threshold_proxy_eval": {"val_proxy_total_false_count": 20.0},
    }

    assert train._halo_selection_tuple(better_halo) > train._halo_selection_tuple(incumbent)
    assert train._halo_selection_tuple(better_false) > train._halo_selection_tuple(incumbent)
    assert train._halo_selection_tuple(better_count) > train._halo_selection_tuple(incumbent)
    assert train._halo_selection_tuple(better_timing) > train._halo_selection_tuple(incumbent)


def test_validate_fixed_stage0_checkpoint_accepts_prod_reduced_rf_halo16() -> None:
    train._validate_fixed_stage0_checkpoint(_prod_checkpoint_payload())


def test_validate_fixed_stage0_checkpoint_rejects_wrong_dilations() -> None:
    try:
        train._validate_fixed_stage0_checkpoint(_prod_checkpoint_payload(dilations=(1, 2, 4, 8)))
    except RuntimeError as exc:
        assert "model_cfg.dilations" in str(exc)
    else:
        raise AssertionError("Expected dilations mismatch to fail")


def test_build_prod_phase1_args_uses_prod_defaults(tmp_path) -> None:
    record = _fake_record(tmp_path)
    cache_spec = train.CacheSpec(
        cache_root=tmp_path / "cache",
        manifest={"split_policy": train.DEFAULT_SPLIT_POLICY, "val_ratio": train.VAL_RATIO},
        train_records=(record,),
        val_eval_records=(record,),
        pooler_path=Path(record.pooler_checkpoint),
        encoder_model="large",
        encoder_checkpoint=tmp_path / "vitl.pt",
        train_segments_path=tmp_path / "cache" / "index" / "train_segments.jsonl",
        val_eval_segments_path=tmp_path / "cache" / "index" / "val_eval_segments.jsonl",
    )
    args = train._build_prod_phase1_args(
        stage0_wrapper_root=tmp_path / "wrapper",
        cache_spec=cache_spec,
        output_root=tmp_path / "out",
        metadata_stamp_path=tmp_path / "wrapper" / "metadata.json",
        device="cuda",
    )

    assert "--variant" in args
    assert args[args.index("--variant") + 1] == "baseline"
    assert args[args.index("--probe-lr") + 1] == str(float(train.PROD_PROBE_LR))
    assert args[args.index("--train-segments-file") + 1] == str(cache_spec.train_segments_path)
    assert args[args.index("--val-segments-file") + 1] == str(cache_spec.val_eval_segments_path)
    assert args[args.index("--halo") + 1] == str(int(train.PROD_HALO))


def test_extract_best_metrics_reads_prod_summary() -> None:
    metrics = train._extract_best_metrics(_prod_summary_payload())
    assert metrics["best_epoch"] == 4
    assert metrics["baseline_halo16_pair_f1"] == 0.915
    assert metrics["val_halo16_pair_f1"] == 0.952
    assert metrics["val_legacy_pair_f1"] == 0.94
    assert metrics["val_chunk32_pair_f1"] == 0.933
    assert metrics["val_proxy_macro_f1"] == 0.901
    assert metrics["val_proxy_false"] == 17


def test_main_writes_run_summary_from_prod_summary(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    record = _fake_record(tmp_path)
    encoder_checkpoint = tmp_path / "vitl.pt"
    encoder_checkpoint.write_bytes(b"encoder")
    cache_spec = train.CacheSpec(
        cache_root=cache_root,
        manifest={"split_policy": train.DEFAULT_SPLIT_POLICY, "val_ratio": train.VAL_RATIO},
        train_records=(record,),
        val_eval_records=(record,),
        pooler_path=Path(record.pooler_checkpoint),
        encoder_model="large",
        encoder_checkpoint=encoder_checkpoint,
        train_segments_path=cache_root / "index" / "train_segments.jsonl",
        val_eval_segments_path=cache_root / "index" / "val_eval_segments.jsonl",
    )
    cache_spec.train_segments_path.parent.mkdir(parents=True, exist_ok=True)
    cache_spec.train_segments_path.write_text("", encoding="utf-8")
    cache_spec.val_eval_segments_path.write_text("", encoding="utf-8")

    fixed_stage0_checkpoint = tmp_path / "fixed_stage0" / "boundary_model.pt"
    fixed_stage0_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_prod_checkpoint_payload(), fixed_stage0_checkpoint)

    monkeypatch.setattr(train, "_resolve_cache_root", lambda: cache_root)
    monkeypatch.setattr(train, "_load_cache_spec", lambda root: cache_spec)
    monkeypatch.setattr(train, "_resolve_fixed_stage0_checkpoint", lambda root: fixed_stage0_checkpoint)
    monkeypatch.setattr(train, "_resolve_output_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(train.time, "strftime", lambda fmt, ts=None: "20260415T000000Z")
    monkeypatch.setattr(train, "_peak_vram_mb", lambda: 0.0)

    def fake_run_prod_phase1_epoch_eval(*, args: list[str]) -> int:
        output_root = Path(args[args.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(
            json.dumps(_prod_summary_payload(), indent=2),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(train, "_run_prod_phase1_epoch_eval", fake_run_prod_phase1_epoch_eval)

    rc = train.main()
    assert rc == 0

    run_root = tmp_path / "runs" / "mangalam_phase1_prod6040_20260415T000000Z"
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["cache_version"] == train.CACHE_VERSION
    assert run_summary["fixed_stage0_checkpoint"] == str(fixed_stage0_checkpoint)
    assert run_summary["best_metrics"]["val_halo16_pair_f1"] == 0.952
    assert run_summary["best_metrics"]["val_proxy_false"] == 17
