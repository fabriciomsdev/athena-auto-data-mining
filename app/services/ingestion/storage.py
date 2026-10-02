from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import structlog
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = structlog.get_logger()


def get_minio_client() -> Minio:
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


class StorageService:
    """Thin wrapper around MinIO for AthenaMining file operations."""

    def __init__(self, client: Minio | None = None) -> None:
        self.client = client or get_minio_client()
        self._ensure_buckets()

    def _ensure_buckets(self) -> None:
        for bucket in [settings.MINIO_RAW_BUCKET, settings.MINIO_MODELS_BUCKET]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info("Created MinIO bucket", bucket=bucket)

    def upload_file(
        self,
        local_path: str | Path,
        bucket: str | None = None,
        object_name: str | None = None,
    ) -> str:
        """Upload a local file and return the MinIO object path."""
        bucket = bucket or settings.MINIO_RAW_BUCKET
        local_path = Path(local_path)
        object_name = object_name or f"{uuid4()}/{local_path.name}"

        self.client.fput_object(bucket, object_name, str(local_path))
        minio_path = f"{bucket}/{object_name}"
        logger.info("Uploaded file to MinIO", path=minio_path)
        return minio_path

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> str:
        """Upload raw bytes to MinIO."""
        bucket = bucket or settings.MINIO_RAW_BUCKET
        self.client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        minio_path = f"{bucket}/{object_name}"
        logger.info("Uploaded bytes to MinIO", path=minio_path, size=len(data))
        return minio_path

    def download_to_temp(self, minio_path: str, dest_dir: str | Path) -> Path:
        """Download a MinIO object to a local temp directory."""
        bucket, *parts = minio_path.split("/", 1)
        object_name = parts[0]
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(object_name).name
        local_path = dest_dir / filename

        self.client.fget_object(bucket, object_name, str(local_path))
        logger.info("Downloaded from MinIO", path=minio_path, dest=str(local_path))
        return local_path

    def get_presigned_url(self, minio_path: str, expires_seconds: int = 3600) -> str:
        """Generate a presigned download URL."""
        from datetime import timedelta

        bucket, *parts = minio_path.split("/", 1)
        object_name = parts[0]
        url = self.client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )
        return url
