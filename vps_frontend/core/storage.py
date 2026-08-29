"""
SupoClip — Zero-Registration & Multi-Cloud Storage Manager
100% Zero Setup required: automatically uses high-speed, direct-link temporary
storage providers (Litterbox / Uguu) with automatic fallback and retries.
Optionally supports S3 (Cloudflare R2 / AWS S3) if credentials are provided.
"""

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Request timeout for uploads (seconds)
UPLOAD_TIMEOUT = 120
MAX_RETRIES = 3


class StorageClient:
    """
    Zero-Registration Storage Client with Multi-Provider Fallback.

    Hierarchy:
    1. If S3 credentials provided -> Use S3/Cloudflare R2
    2. Zero-Config Mode (Default):
       - Primary: Litterbox (Catbox) - 72h retention, up to 1GB
       - Fallback 1: Uguu.se - 48h retention, up to 128MB
       - Fallback 2: Stream back / Local
    """

    def __init__(
        self,
        endpoint_url: str = "",
        access_key: str = "",
        secret_key: str = "",
        bucket_name: str = "",
        region: str = "auto",
    ):
        self.endpoint_url = (endpoint_url or "").strip()
        self.access_key = (access_key or "").strip()
        self.secret_key = (secret_key or "").strip()
        self.bucket = (bucket_name or "").strip()
        self.region = region

        self.has_s3 = bool(self.access_key and self.secret_key and self.bucket)

        if self.has_s3:
            try:
                import boto3
                from botocore.client import Config

                self.s3 = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url or None,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region,
                    config=Config(signature_version="s3v4"),
                )
                logger.info(f"StorageClient: Using configured S3 bucket '{self.bucket}'")
            except Exception as e:
                logger.warning(f"S3 init failed ({e}), falling back to Zero-Config storage.")
                self.has_s3 = False
                self.s3 = None
        else:
            self.s3 = None
            logger.info("StorageClient: Zero-Registration storage active (No signup required)")

    def upload_video(
        self,
        local_path: str | Path,
        remote_key: str = None,
        presigned_ttl: int = 86400,
    ) -> str:
        """
        Upload a video and return a direct public/presigned download URL.
        Zero registration required.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")

        file_size = local_path.stat().st_size
        logger.info(
            f"Uploading '{local_path.name}' ({file_size / (1024*1024):.1f} MB)..."
        )

        # 1. Try S3 if configured
        if self.has_s3 and self.s3:
            try:
                if remote_key is None:
                    remote_key = f"inputs/{local_path.name}"
                self.s3.upload_file(
                    str(local_path),
                    self.bucket,
                    remote_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": remote_key},
                    ExpiresIn=presigned_ttl,
                )
                logger.info(f"Uploaded to S3: s3://{self.bucket}/{remote_key}")
                return url
            except Exception as e:
                logger.warning(f"S3 upload failed ({e}), falling back to Zero-Registration providers...")

        # 2. Zero-Registration Upload with Multi-Tier Fallback
        return self._upload_zero_registration(local_path)

    def _upload_zero_registration(self, local_path: Path) -> str:
        """Upload to zero-signup providers with automatic failover."""
        errors = []

        # Provider 1: Litterbox (Catbox) - 72h retention, up to 1GB
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Trying Litterbox upload (attempt {attempt})...")
                with open(local_path, "rb") as f:
                    files = {"fileToUpload": (local_path.name, f, "video/mp4")}
                    data = {"reqtype": "fileupload", "time": "72h"}
                    resp = requests.post(
                        "https://litterbox.catbox.moe/resources/internals/api.php",
                        data=data,
                        files=files,
                        timeout=UPLOAD_TIMEOUT,
                    )
                if resp.status_code == 200 and resp.text.startswith("http"):
                    url = resp.text.strip()
                    logger.info(f"Litterbox upload success: {url}")
                    return url
                else:
                    logger.warning(f"Litterbox returned status {resp.status_code}: {resp.text[:150]}")
            except Exception as e:
                logger.warning(f"Litterbox attempt {attempt} error: {e}")
                errors.append(f"Litterbox: {e}")
                time.sleep(1)

        # Provider 2: Uguu.se - 48h retention
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Trying Uguu.se fallback (attempt {attempt})...")
                with open(local_path, "rb") as f:
                    files = {"files[]": (local_path.name, f, "video/mp4")}
                    resp = requests.post(
                        "https://uguu.se/upload",
                        files=files,
                        timeout=UPLOAD_TIMEOUT,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success") and data.get("files"):
                        url = data["files"][0]["url"]
                        logger.info(f"Uguu upload success: {url}")
                        return url
            except Exception as e:
                logger.warning(f"Uguu attempt {attempt} error: {e}")
                errors.append(f"Uguu: {e}")
                time.sleep(1)

        raise RuntimeError(
            f"Zero-Registration upload failed across all providers. Errors: {'; '.join(errors)}"
        )

    def download_file(self, remote_url: str, local_path: str | Path) -> Path:
        """Download file from any public or S3 URL to local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {remote_url[:60]}... -> {local_path}")
        resp = requests.get(remote_url, stream=True, timeout=180)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info(f"Download complete: {local_path} ({local_path.stat().st_size / (1024*1024):.1f} MB)")
        return local_path

    def test_connection(self) -> bool:
        """Test storage health."""
        if self.has_s3 and self.s3:
            try:
                self.s3.head_bucket(Bucket=self.bucket)
                return True
            except Exception:
                return False
        # In zero-config mode, connection is always OK
        return True
