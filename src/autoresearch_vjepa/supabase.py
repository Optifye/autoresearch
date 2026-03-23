"""
Supabase client wrapper mirroring the legacy pipeline behaviour.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar
from urllib.parse import urlparse, urlunparse

try:
    from supabase import Client, create_client
except ImportError as exc:  # pragma: no cover - environment validation
    Client = object  # type: ignore
    create_client = None
    _SUPABASE_IMPORT_ERROR = exc
else:
    _SUPABASE_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)
_TRANSIENT_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_T = TypeVar("_T")


def _normalise_supabase_url(raw_url: str) -> str:
    """Normalise potentially quoted Supabase URLs and ensure a scheme is present."""
    candidate = raw_url.strip().strip("\"'")  # tolerate dotenv quoting
    if not candidate:
        raise ValueError("Supabase URL is empty.")
    if any(ch.isspace() for ch in candidate):
        raise ValueError(f"Supabase URL contains whitespace: {raw_url!r}")

    if "://" not in candidate:
        # Default to https unless a scheme is explicitly provided.
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Supabase URL appears malformed: {raw_url!r}")

    # Drop trailing slash to match expectations of supabase-python.
    path = parsed.path.rstrip("/")
    normalised = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return normalised


def _parse_supabase_error_payload(exc: Exception) -> Optional[Dict[str, Any]]:
    payload = exc.args[0] if exc.args else None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        raw = payload.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, dict):
                return parsed
    return None


def _extract_supabase_status_code(exc: Exception) -> Optional[int]:
    payload = _parse_supabase_error_payload(exc)
    if isinstance(payload, dict):
        raw_code = payload.get("code")
        if isinstance(raw_code, int):
            return raw_code
        if isinstance(raw_code, str):
            stripped = raw_code.strip()
            if stripped.isdigit():
                return int(stripped)
    text = str(exc)
    for code in _TRANSIENT_HTTP_STATUS_CODES:
        if f" {code} " in text or f"={code}" in text or f": {code}" in text or f"'{code}'" in text:
            return code
    return None


def _is_transient_supabase_error(exc: Exception) -> bool:
    status_code = _extract_supabase_status_code(exc)
    if status_code in _TRANSIENT_HTTP_STATUS_CODES:
        return True
    text = str(exc).lower()
    transient_markers = (
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "too many requests",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
    )
    return any(marker in text for marker in transient_markers)


def _summarise_exception(exc: Exception, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


class EmbeddingDBClient:
    """Client for interacting with Supabase tables used by the pipeline."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> None:
        if create_client is None:
            raise ImportError(
                "supabase python client is required. Install via `pip install supabase`."
            ) from _SUPABASE_IMPORT_ERROR
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY "
                "(service role) or SUPABASE_SERVICE_ROLE_KEY."
            )
        cleaned_url = _normalise_supabase_url(url)
        self._skip_updates = os.getenv("SUPABASE_SKIP_UPDATES", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._retry_attempts = max(1, int(os.getenv("SUPABASE_RETRY_ATTEMPTS", "4")))
        self._retry_base_delay_seconds = max(0.0, float(os.getenv("SUPABASE_RETRY_DELAY_SECONDS", "1.0")))
        try:
            self.client: Client = create_client(cleaned_url, key)
            print(f"✓ Connected to Supabase project: {cleaned_url}")
            if self._skip_updates:
                print("Supabase updates disabled (SUPABASE_SKIP_UPDATES=1).")
        except Exception as exc:
            raise ValueError(
                "Failed to initialise Supabase client. Double-check SUPABASE_URL and "
                "ensure SUPABASE_KEY is the service role key (Settings → API → service_role)."
            ) from exc

    def _skip_write(self, action: str) -> bool:
        if not self._skip_updates:
            return False
        print(f"[supabase] Skipping {action} update (SUPABASE_SKIP_UPDATES=1).")
        return True

    def _execute_with_retry(self, operation: Callable[[], _T], *, action: str) -> _T:
        attempts = max(1, int(getattr(self, "_retry_attempts", 1)))
        base_delay = max(0.0, float(getattr(self, "_retry_base_delay_seconds", 0.0)))
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= attempts or not _is_transient_supabase_error(exc):
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Transient Supabase error during %s (attempt %d/%d): %s; retrying in %.1fs",
                    action,
                    attempt,
                    attempts,
                    _summarise_exception(exc),
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"unreachable retry state for {action}")

    # ------------------------------------------------------------------ fetch
    def get_embedding_run(self, run_id: str) -> Dict:
        result = self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_runs")
                .select("*")
                .eq("id", run_id)
                .execute()
            ),
            action="get_embedding_run",
        )
        if not result.data:
            raise ValueError(f"Embedding run {run_id} not found.")
        run = result.data[0]
        if isinstance(run.get("config_snapshot"), str):
            run["config_snapshot"] = json.loads(run["config_snapshot"])
        if isinstance(run.get("clips_per_class"), str):
            run["clips_per_class"] = json.loads(run["clips_per_class"])
        return run

    def get_embedding_space(self, space_id: str) -> Dict:
        result = self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_spaces")
                .select("*")
                .eq("id", space_id)
                .execute()
            ),
            action="get_embedding_space",
        )
        if not result.data:
            raise ValueError(f"Embedding space {space_id} not found.")
        space = result.data[0]
        for key in ("roi_default_config", "model_default_config"):
            if isinstance(space.get(key), str):
                space[key] = json.loads(space[key])
        return space

    # --------------------------------------------------------------- mutators
    def update_embedding_run_status(self, run_id: str, status: str) -> None:
        if self._skip_write("embedding_run_status"):
            return
        self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_runs")
                .update({"status": status})
                .eq("id", run_id)
                .execute()
            ),
            action="update_embedding_run_status",
        )

    def update_embedding_run_results(
        self,
        run_id: str,
        embeddings_s3_path: Optional[str],
        size_bytes: int,
        classification_accuracy: Optional[float],
        confusion_matrix: Optional[dict] = None,
        processing_duration: Optional[int] = None,
    ) -> None:
        if self._skip_write("embedding_run_results"):
            return
        import datetime

        payload = {
            "status": "completed",
            "embeddings_npz_s3_path": embeddings_s3_path,
            "embeddings_npz_size_bytes": size_bytes,
            "classification_accuracy": classification_accuracy,
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": None,
            "error_details": None,
        }
        if confusion_matrix:
            payload["confusion_matrix"] = json.dumps(confusion_matrix)
        if processing_duration is not None:
            payload["processing_duration_seconds"] = processing_duration
        self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_runs")
                .update(payload)
                .eq("id", run_id)
                .execute()
            ),
            action="update_embedding_run_results",
        )

    def update_embedding_run_error(
        self,
        run_id: str,
        error_message: str,
        error_details: Optional[dict] = None,
    ) -> None:
        if self._skip_write("embedding_run_error"):
            return
        import datetime

        payload = {
            "status": "failed",
            "error_message": error_message,
            "completed_at": datetime.datetime.utcnow().isoformat(),
        }
        if error_details:
            payload["error_details"] = json.dumps(error_details)
        self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_runs")
                .update(payload)
                .eq("id", run_id)
                .execute()
            ),
            action="update_embedding_run_error",
        )

    def update_embedding_space_latest_run(self, space_id: str, run_id: str) -> None:
        if self._skip_write("embedding_space_latest_run"):
            return
        result = self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_spaces")
                .select("total_runs")
                .eq("id", space_id)
                .execute()
            ),
            action="update_embedding_space_latest_run.select_total_runs",
        )
        total_runs = result.data[0]["total_runs"] + 1 if result.data else 1
        self._execute_with_retry(
            lambda: (
                self.client.schema("cv")
                .table("embedding_spaces")
                .update({"latest_embedding_run_id": run_id, "total_runs": total_runs})
                .eq("id", space_id)
                .execute()
            ),
            action="update_embedding_space_latest_run.update",
        )

    def update_camera_models(
        self,
        space_id: str,
        run_id: str,
        run_number: int,
        embeddings_s3_path: str,
        accuracy: Optional[float],
        total_clips: int,
        camera_artifacts: Optional[Dict[str, str]] = None,
        camera_model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if self._skip_write("camera_models"):
            return
        import datetime

        space = self.get_embedding_space(space_id)
        camera_ids = space.get("assigned_camera_ids", [])
        if not camera_ids:
            return
        print(f"[update_camera_models] assigned_camera_ids={camera_ids}")

        for camera_id in camera_ids:
            try:
                camera_result = self._execute_with_retry(
                    lambda: (
                        self.client.schema("public")
                        .table("cameras")
                        .select("id,camera_id,model_id,model_name")
                        .eq("id", camera_id)
                        .execute()
                    ),
                    action=f"update_camera_models.select_camera.{camera_id}",
                )
                if not camera_result.data:
                    print(f"[update_camera_models] skipping camera {camera_id}: no row found in public.cameras")
                    continue
                camera = camera_result.data[0]
                model_id = camera.get("model_id")
                if not model_id:
                    print(f"[update_camera_models] skipping camera {camera_id}: model_id missing")
                    continue

                artifact_path = (
                    camera_artifacts.get(camera_id)
                    if camera_artifacts and camera_id in camera_artifacts
                    else embeddings_s3_path
                )
                print(
                    f"[update_camera_models] updating model {model_id} for camera {camera_id} "
                    f"(camera_id_in_row={camera.get('camera_id')}) with artifact {artifact_path}"
                )
                model_config = {
                    "embedding_run": {
                        "run_id": run_id,
                        "run_number": run_number,
                        "space_id": space_id,
                        "embeddings_s3_path": artifact_path,
                        "accuracy": accuracy,
                        "total_clips": total_clips,
                        "updated_at": datetime.datetime.utcnow().isoformat(),
                    }
                }
                camera_override = (
                    camera_model_configs.get(camera_id)
                    if camera_model_configs and camera_id in camera_model_configs
                    else None
                )
                if camera_override:
                    model_config = _deep_merge_dicts(model_config, _coerce_json_dict(camera_override))
                existing_model_response = self._execute_with_retry(
                    lambda: (
                        self.client.schema("public")
                        .table("models")
                        .select("model_configuration")
                        .eq("id", model_id)
                        .execute()
                    ),
                    action=f"update_camera_models.select_model.{model_id}",
                )
                existing_model_config = {}
                if existing_model_response.data:
                    existing_model_config = _coerce_json_dict(
                        existing_model_response.data[0].get("model_configuration")
                    )
                merged_config = _deep_merge_dicts(existing_model_config, model_config)
                update_payload = {
                    "model_path": artifact_path,
                    "model_configuration": json.dumps(merged_config),
                    "updated_at": datetime.datetime.utcnow().isoformat(),
                    # Camera-linked shared models can sit in a disabled placeholder state.
                    # Re-enable them before updating or the model_not_in_use() check
                    # constraint rejects the write while cameras still reference the row.
                    "enabled": True,
                }
                self._execute_with_retry(
                    lambda: (
                        self.client.schema("public")
                        .table("models")
                        .update(update_payload)
                        .eq("id", model_id)
                        .execute()
                    ),
                    action=f"update_camera_models.update_model.{model_id}",
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Failed to update model for camera {camera_id}: {exc}")


def _coerce_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _deep_merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged
