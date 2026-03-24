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


def test_stage_budget_split_scales_total_budget(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TCN_STAGE_SECONDS", raising=False)
    monkeypatch.delenv("AUTORESEARCH_PROBE_STAGE_SECONDS", raising=False)

    tcn_default, probe_default = train.resolve_stage_budget_seconds(600.0)
    assert tcn_default == 300.0
    assert probe_default == 300.0

    tcn_scaled, probe_scaled = train.resolve_stage_budget_seconds(10.0)
    assert tcn_scaled == 5.0
    assert probe_scaled == 5.0

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
