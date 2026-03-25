from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import prepare
from autoresearch_vjepa import cache_contract
from autoresearch_vjepa import materialize


def _write_feature(path: Path, *, video_id: str, camera_id: str, fps: float) -> None:
    timestamps_ms = np.arange(6, dtype=np.int64) * 1000
    embeddings = np.arange(18, dtype=np.float32).reshape(6, 3)
    tokens = np.arange(6 * 4 * 3, dtype=np.float16).reshape(6, 4, 3)
    np.savez(
        path,
        tokens=tokens,
        embeddings=embeddings,
        timestamps_ms=timestamps_ms,
        embedding_kind=np.array("base"),
        model_name=np.array("ssv2"),
        clip_len=np.array(16),
        stride=np.array(10),
        frame_skip=np.array(0),
        timestamp_alignment=np.array("center"),
        pooler_sha=np.array("pooler_sha"),
        camera_id=np.array(camera_id),
        video_id=np.array(video_id),
        fps=np.array(fps),
    )


def _write_label(path: Path, *, video_id: str, camera_id: str, fps: float) -> None:
    payload = {
        "video_id": video_id,
        "camera_id": camera_id,
        "fps": fps,
        "supervised_start_ms": 0,
        "supervised_end_ms": 5000,
        "cycles": [
            {"start_ms": 1000, "end_ms": 2000},
            {"start_ms": 3000, "end_ms": 4000},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root: Path, *, video_id: str, camera_id: str, raw_path: str) -> Path:
    run_root = root / f"run_{video_id}" / "dense_temporal"
    features_dir = run_root / "features"
    labels_dir = run_root / "labels"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    _write_feature(features_dir / f"{video_id}__features.npz", video_id=video_id, camera_id=camera_id, fps=25.0)
    _write_label(labels_dir / f"{video_id}__seg000.json", video_id=video_id, camera_id=camera_id, fps=25.0)
    pooler_path = root / f"{video_id}_pooler.pt"
    pooler_path.write_bytes(b"pooler")
    (run_root / "resolved_config.json").write_text(
        json.dumps({"model": {"pooler_path": str(pooler_path)}}),
        encoding="utf-8",
    )
    (run_root / "snapshot.json").write_text(
        json.dumps({"videos": [{"video_id": video_id, "camera_id": camera_id, "path": raw_path}]}),
        encoding="utf-8",
    )
    return run_root


def test_build_cache_and_memmap(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    cache_contract.configure_cache_paths(cache_root)
    monkeypatch.setattr(
        cache_contract,
        "_stable_fraction",
        lambda key, seed: 0.1 if "val" in key else 0.9,
    )

    run_train = _write_run(tmp_path, video_id="video_train", camera_id="camera_a", raw_path="s3://bucket/train.mp4")
    run_val = _write_run(tmp_path, video_id="video_val", camera_id="camera_a", raw_path="s3://bucket/val.mp4")
    summary = prepare.build_cache(
        source_run_dirs=[str(run_train), str(run_val)],
        source_globs=[],
        camera_include_regex=None,
        video_include_regex=None,
        path_include_regex=None,
        split_policy="camera_stratified_hash",
        val_ratio=0.5,
        seed=42,
        force=False,
    )

    assert summary["train_segments"] == 1
    assert summary["val_segments"] == 1
    assert summary["val_eval_segments"] == 1

    train_records = prepare.load_split_records("train", cache_root=cache_root)
    val_records = prepare.load_split_records("val_eval", cache_root=cache_root)
    assert len(train_records) == 1
    assert len(val_records) == 1

    payload = prepare.load_segment_arrays(train_records[0], representation="both", use_eval_span=False)
    assert payload["pooled_z0"].shape == (6, 3)
    assert payload["tokens"].shape == (6, 4, 3)

    tokens_mm = prepare.memmap_npz_member(Path(train_records[0].feature_path), "tokens")
    assert tokens_mm.shape == (6, 4, 3)
    assert float(tokens_mm[1, 0, 0]) == 12.0


def test_build_cache_camera_stratified_hash_split_handles_singleton_cameras(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_contract.configure_cache_paths(cache_root)
    run_specs = [
        ("v1", "cam_a"),
        ("v2", "cam_a"),
        ("v3", "cam_b"),
        ("v4", "cam_b"),
        ("v5", "cam_c"),
    ]
    for video_id, camera_id in run_specs:
        _write_run(tmp_path, video_id=video_id, camera_id=camera_id, raw_path=f"s3://bucket/{video_id}.mp4")

    summary = prepare.build_cache(
        source_run_dirs=[str(tmp_path / f"run_{video_id}" / "dense_temporal") for video_id, _ in run_specs],
        source_globs=[],
        camera_include_regex=None,
        video_include_regex=None,
        path_include_regex=None,
        split_policy="camera_stratified_hash",
        val_ratio=0.4,
        seed=42,
        force=False,
    )

    manifest = prepare.load_manifest(cache_root=cache_root)
    train_records = prepare.load_split_records("train", cache_root=cache_root)
    val_records = prepare.load_split_records("val", cache_root=cache_root)
    val_eval_records = prepare.load_split_records("val_eval", cache_root=cache_root)

    assert summary["split_policy"] == "camera_stratified_hash"
    assert manifest["split_policy"] == "camera_stratified_hash"
    assert manifest["camera_total_counts"] == {"cam_a": 2, "cam_b": 2, "cam_c": 1}
    assert manifest["camera_val_counts"]["cam_a"] == 1
    assert manifest["camera_val_counts"]["cam_b"] == 1
    assert manifest["camera_val_counts"]["cam_c"] == 0
    assert manifest["split_target_val_videos"] == 2.0
    assert len({record.video_id for record in train_records}) == 3
    assert len({record.video_id for record in val_records}) == 2
    assert len(val_eval_records) == len(val_records)


def test_pair_matching_and_metrics() -> None:
    record = prepare.SegmentRecord(
        segment_id="seg0",
        split="val",
        video_id="vid",
        camera_id="cam",
        source_run_dir="/tmp/run",
        feature_path="/tmp/fake.npz",
        label_path="/tmp/fake.json",
        pooler_checkpoint="",
        pooler_sha="sha",
        embedding_dim=3,
        token_dim=3,
        tokens_per_window=4,
        num_total_windows=6,
        fps=25.0,
        supervised_start_ms=0,
        supervised_end_ms=5000,
        supervised_start_idx=0,
        supervised_end_idx=5,
        eval_start_ms=0,
        eval_end_ms=5000,
        eval_start_idx=0,
        eval_end_idx=5,
        event_pairs_ms=((1000, 2000), (3000, 4000)),
    )
    predictions = {
        "seg0": [
            prepare.EventPair(start_ms=1050, end_ms=1950),
            prepare.EventPair(start_ms=2900, end_ms=3950),
            prepare.EventPair(start_ms=4500, end_ms=4700),
        ]
    }
    metrics = prepare.evaluate_predictions(predictions, records=[record])
    assert metrics["val_pair_precision"] == 2 / 3
    assert metrics["val_pair_recall"] == 1.0
    assert round(metrics["val_pair_f1"], 6) == round(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0), 6)
    assert metrics["val_count_mae"] == 1.0
    assert metrics["val_start_mae_ms"] == 75.0
    assert metrics["val_end_mae_ms"] == 50.0


def test_decode_config_is_zero_gap() -> None:
    cfg = prepare.DEFAULT_DECODE_CONFIG
    assert cfg.peak_min_distance_s == 0.0
    assert cfg.min_pair_s == 0.0
    assert cfg.min_gap_s == 0.0


def test_snapshot_path_materializes_without_run_id(tmp_path, monkeypatch) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text("{}", encoding="utf-8")

    calls: list[tuple[str, str | None, int | None, str | None]] = []

    def fake_materialize_run_source(
        *,
        run_id: str,
        space_id: str | None,
        run_number: int | None,
        snapshot_path: Path | None,
        force_reextract: bool,
        camera_include_regex: str | None = None,
        video_include_regex: str | None = None,
        path_include_regex: str | None = None,
    ) -> Path:
        del force_reextract, camera_include_regex, video_include_regex, path_include_regex
        calls.append((run_id, space_id, run_number, str(snapshot_path) if snapshot_path is not None else None))
        materialized = tmp_path / "materialized" / "dense_temporal"
        materialized.mkdir(parents=True, exist_ok=True)
        return materialized

    monkeypatch.setattr(cache_contract, "materialize_run_source", fake_materialize_run_source)

    resolved = cache_contract.resolve_requested_source_dirs(
        source_run_dirs=[],
        run_id=None,
        space_id="demo-space",
        run_number=7,
        snapshot_path=snapshot_path,
        force_materialize=True,
    )

    assert resolved == [str((tmp_path / "materialized" / "dense_temporal").resolve())]
    assert len(calls) == 1
    assert calls[0][0].startswith("snapshot__demo-space__7__")
    assert calls[0][1:] == ("demo-space", 7, str(snapshot_path))


def test_discover_reusable_feature_paths_matches_existing_source(tmp_path, monkeypatch) -> None:
    reuse_root = tmp_path / "old_source" / "dense_temporal"
    features_dir = reuse_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_path = features_dir / "vid0__features.npz"
    _write_feature(feature_path, video_id="vid0", camera_id="cam0", fps=25.0)
    (reuse_root / "resolved_config.json").write_text(
        json.dumps(
            {
                "space_id": "space-123",
                "model": {"encoder_model": "large", "preproc_id": "vjepa_rgb_256"},
                "train": {"clip_len": 16, "stride": 10, "frame_skip": 0},
            }
        ),
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        space_id="space-123",
        model=SimpleNamespace(encoder_model="large", preproc_id="vjepa_rgb_256"),
        train=SimpleNamespace(clip_len=16, stride=10, frame_skip=0),
        videos=(SimpleNamespace(video_id="vid0", camera_id="cam0"), SimpleNamespace(video_id="vid1", camera_id="cam1")),
    )
    monkeypatch.setenv("AUTORESEARCH_REUSE_SOURCE_RUN_DIRS", str(reuse_root))
    monkeypatch.delenv("AUTORESEARCH_DISABLE_FEATURE_REUSE", raising=False)

    reusable = materialize._discover_reusable_feature_paths(
        cfg=cfg,
        work_dir=tmp_path / "new_source" / "dense_temporal",
        pooler_sha="pooler_sha",
    )

    assert reusable == {"vid0": feature_path.resolve()}
