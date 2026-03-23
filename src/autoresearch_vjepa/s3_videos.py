"""
S3 video download and caching utilities mirrored from the legacy pipeline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import logging
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class ArchivedS3ObjectError(RuntimeError):
    def __init__(self, s3_path: str, *, storage_class: Optional[str] = None, restore: Optional[str] = None) -> None:
        self.s3_path = s3_path
        self.storage_class = storage_class
        self.restore = restore
        detail = []
        if storage_class:
            detail.append(f"storage_class={storage_class}")
        if restore:
            detail.append(f"restore={restore}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        super().__init__(f"Archived S3 object unavailable: {s3_path}{suffix}")


class S3VideoDownloader:
    """Download and cache videos locally for clip extraction."""

    def __init__(self, cache_dir: str, region_name: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        max_pool = int(os.getenv("S3_MAX_POOL_CONNECTIONS", "64") or "64")
        self.s3_client = boto3.client("s3", region_name=region_name, config=Config(max_pool_connections=max_pool))

    @staticmethod
    def is_s3_path(path: str) -> bool:
        return path.startswith("s3://")

    @staticmethod
    def parse_s3_path(path: str) -> tuple[str, str]:
        parsed = urlparse(path)
        if parsed.scheme != "s3":
            raise ValueError(f"Invalid S3 path: {path}")
        return parsed.netloc, parsed.path.lstrip("/")

    def get_cache_path(self, s3_path: str) -> Path:
        bucket, key = self.parse_s3_path(s3_path)
        digest = hashlib.md5(s3_path.encode("utf-8")).hexdigest()[:12]
        name = Path(key).name
        stem, suffix = Path(name).stem, Path(name).suffix
        return self.cache_dir / f"{stem}_{digest}{suffix}"

    def describe_object(self, s3_path: str) -> dict:
        bucket, key = self.parse_s3_path(s3_path)
        try:
            head = self.s3_client.head_object(Bucket=bucket, Key=key)
        except Exception:
            return {}
        return {
            "storage_class": head.get("StorageClass"),
            "restore": head.get("Restore"),
            "size": head.get("ContentLength"),
        }

    def request_restore(self, s3_path: str, *, days: int, tier: str) -> bool:
        bucket, key = self.parse_s3_path(s3_path)
        request = {"Days": int(days), "GlacierJobParameters": {"Tier": tier}}
        self.s3_client.restore_object(Bucket=bucket, Key=key, RestoreRequest=request)
        return True

    def download(self, s3_path: str, *, force: bool = False) -> Path:
        cache_path = self.get_cache_path(s3_path)
        if cache_path.exists() and not force:
            return cache_path
        bucket, key = self.parse_s3_path(s3_path)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(bucket, key, str(cache_path))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code == "InvalidObjectState":
                details = self.describe_object(s3_path)
                storage_class = details.get("storage_class")
                restore = details.get("restore")
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
                raise ArchivedS3ObjectError(s3_path, storage_class=storage_class, restore=restore) from exc
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
            raise
        except Exception:
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
            raise
        return cache_path

    def get_video_path(self, path: str, *, force: bool = False) -> str:
        if not self.is_s3_path(path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Video not found: {path}")
            return path
        return str(self.download(path, force=force))


class R2VideoDownloader:
    """Download and cache videos from Cloudflare R2 (S3-compatible)."""

    def __init__(
        self,
        cache_dir: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region_name: Optional[str] = None,
        fallback_endpoint_url: Optional[str] = None,
        fallback_access_key: Optional[str] = None,
        fallback_secret_key: Optional[str] = None,
        fallback_region_name: Optional[str] = None,
        fallback_on_not_found: Optional[bool] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

        endpoint = endpoint_url or os.getenv("R2_S3_ENDPOINT_URL")
        access_key = access_key or os.getenv("R2_ACCESS_KEY")
        secret_key = secret_key or os.getenv("R2_SECRET_KEY")
        region_name = region_name or os.getenv("R2_REGION")
        if not endpoint:
            raise ValueError("R2 endpoint not configured. Set R2_S3_ENDPOINT_URL.")
        if not access_key or not secret_key:
            raise ValueError("R2 credentials missing. Set R2_ACCESS_KEY and R2_SECRET_KEY.")

        max_pool = int(os.getenv("S3_MAX_POOL_CONNECTIONS", "64") or "64")
        self.s3_client = self._make_client(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region_name=region_name,
            max_pool=max_pool,
        )
        fb_endpoint = fallback_endpoint_url or os.getenv("R2_FALLBACK_S3_ENDPOINT_URL")
        fb_access = fallback_access_key or os.getenv("R2_FALLBACK_ACCESS_KEY")
        fb_secret = fallback_secret_key or os.getenv("R2_FALLBACK_SECRET_KEY")
        fb_region = fallback_region_name or os.getenv("R2_FALLBACK_REGION")
        self.fallback_client = None
        if fb_endpoint and fb_access and fb_secret:
            self.fallback_client = self._make_client(
                endpoint=fb_endpoint,
                access_key=fb_access,
                secret_key=fb_secret,
                region_name=fb_region,
                max_pool=max_pool,
            )
        if fallback_on_not_found is None:
            raw = os.getenv("R2_FALLBACK_ON_NOT_FOUND", "1")
            fallback_on_not_found = str(raw).strip().lower() not in {"0", "false", "no", "off"}
        self.fallback_on_not_found = bool(fallback_on_not_found)

    @staticmethod
    def _make_client(
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region_name: Optional[str],
        max_pool: int,
    ):
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
            config=Config(max_pool_connections=max_pool),
        )

    @staticmethod
    def is_r2_path(path: str) -> bool:
        return path.startswith("r2://")

    @staticmethod
    def parse_r2_path(path: str) -> tuple[str, str]:
        parsed = urlparse(path)
        if parsed.scheme != "r2":
            raise ValueError(f"Invalid R2 path: {path}")
        return parsed.netloc, parsed.path.lstrip("/")

    def get_cache_path(self, r2_path: str) -> Path:
        bucket, key = self.parse_r2_path(r2_path)
        digest = hashlib.md5(r2_path.encode("utf-8")).hexdigest()[:12]
        name = Path(key).name
        stem, suffix = Path(name).stem, Path(name).suffix
        return self.cache_dir / f"{stem}_{digest}{suffix}"

    def _fallback_reason(self, exc: Exception) -> Optional[str]:
        if not isinstance(exc, ClientError):
            return None
        code = exc.response.get("Error", {}).get("Code")
        if code in {
            "AccessDenied",
            "Forbidden",
            "SignatureDoesNotMatch",
            "InvalidAccessKeyId",
            "AllAccessDisabled",
            "NoSuchBucket",
            "AuthorizationHeaderMalformed",
        }:
            return "forbidden"
        if self.fallback_on_not_found and code in {"NoSuchKey", "NotFound", "404"}:
            return "not_found"
        return None

    def _download_with_client(self, client, bucket: str, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))

    def download(self, r2_path: str, *, force: bool = False) -> Path:
        cache_path = self.get_cache_path(r2_path)
        if cache_path.exists() and not force:
            return cache_path
        bucket, key = self.parse_r2_path(r2_path)
        try:
            self._download_with_client(self.s3_client, bucket, key, cache_path)
        except Exception as exc:
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
            reason = self._fallback_reason(exc)
            if not reason or self.fallback_client is None:
                raise
            try:
                self._download_with_client(self.fallback_client, bucket, key, cache_path)
                self._logger.warning(
                    "R2 fallback download succeeded (reason=%s bucket=%s key=%s)",
                    reason,
                    bucket,
                    key,
                )
            except Exception as fallback_exc:
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
                raise fallback_exc from exc
        return cache_path

    def get_video_path(self, path: str, *, force: bool = False) -> str:
        if not self.is_r2_path(path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Video not found: {path}")
            return path
        return str(self.download(path, force=force))
