"""Utility functions for interacting with S3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from src.pipeline.aws_credentials import get_s3_client
from shutil import copy2


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if uri.startswith("file://"):
        local = uri.split("file://", 1)[1]
        return "", local
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    without_scheme = uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


class S3Downloader:
    """Download S3 prefixes to local storage with progress logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.client = get_s3_client()

    def download_prefix(self, uri: str, dest: Path) -> int:
        if uri.startswith("file://"):
            local_root = Path(uri.split("file://", 1)[1])
            dest.mkdir(parents=True, exist_ok=True)
            total = 0
            if local_root.exists():
                for file_path in sorted(local_root.glob("*.mp4")):
                    target = dest / file_path.name
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copy2(file_path, target)
                    total += 1
            return total

        bucket, prefix = parse_s3_uri(uri)
        dest.mkdir(parents=True, exist_ok=True)
        paginator = self.client.get_paginator("list_objects_v2")

        total_files = 0
        self.logger.info("Listing objects under %s", uri)
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects = page.get("Contents", [])
            total_files += len(objects)
            for obj in objects:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel_path = key[len(prefix) :].lstrip("/")
                if not rel_path:
                    rel_path = Path(key).name
                local_path = dest / rel_path
                if local_path.exists():
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.debug("Downloading %s -> %s", key, local_path)
                self.client.download_file(bucket, key, str(local_path))

        self.logger.info("Downloaded %s: %d files", uri, total_files)
        return total_files

    def download_many(self, mapping: Dict[str, Sequence[str]], base_dir: Path) -> Dict[str, int]:
        """Download multiple S3 prefixes sequentially with logging."""

        stats: Dict[str, int] = {}
        for class_name, uris in mapping.items():
            dest = base_dir / class_name
            total = 0
            if isinstance(uris, str):
                uri_list = [uris]
            else:
                uri_list = list(uris)
            for uri in uri_list:
                count = self.download_prefix(uri, dest)
                total += count
            stats[class_name] = total
        return stats
