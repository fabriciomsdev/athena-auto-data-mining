from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import DataSource, DataSourceType
from app.db.session import get_db
from app.schemas.data_source import DataSourceCreate, DataSourceOut
from app.services.ingestion.storage import StorageService

logger = structlog.get_logger()
router = APIRouter()


ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

MAX_UPLOAD_SIZE_MB = 200


@router.post(
    "/upload",
    response_model=DataSourceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV or Excel file as a data source",
)
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> DataSourceOut:
    # ── Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed: CSV, XLS, XLSX.",
        )

    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    # ── Determine source type ─────────────────────────────────────────────────
    suffix = Path(file.filename or "upload").suffix.lower()
    source_type = DataSourceType.CSV if suffix == ".csv" else DataSourceType.EXCEL

    # ── Upload to MinIO ───────────────────────────────────────────────────────
    storage = StorageService()
    object_name = f"uploads/{Path(file.filename or 'upload').stem}_{Path(file.filename or 'upload').suffix}"
    minio_path = storage.upload_bytes(
        data=content,
        object_name=object_name,
        content_type=file.content_type or "application/octet-stream",
    )

    # ── Persist DataSource record ─────────────────────────────────────────────
    data_source = DataSource(
        name=name,
        source_type=source_type,
        minio_path=minio_path,
        row_count=None,  # populated during data_prep
    )
    db.add(data_source)
    await db.flush()
    await db.refresh(data_source)

    logger.info("DataSource created", id=str(data_source.id), path=minio_path)
    return DataSourceOut.model_validate(data_source)


@router.post(
    "/connect/google-sheets",
    response_model=DataSourceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a Google Sheets URL as a data source",
)
async def connect_google_sheets(
    payload: DataSourceCreate,
    db: AsyncSession = Depends(get_db),
) -> DataSourceOut:
    data_source = DataSource(
        name=payload.name,
        source_type=DataSourceType.GOOGLE_SHEETS,
        external_uri=payload.external_uri,
    )
    db.add(data_source)
    await db.flush()
    await db.refresh(data_source)

    logger.info("Google Sheets DataSource registered", id=str(data_source.id))
    return DataSourceOut.model_validate(data_source)


@router.post(
    "/connect/bigquery",
    response_model=DataSourceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a BigQuery table or SQL query as a data source",
)
async def connect_bigquery(
    payload: DataSourceCreate,
    db: AsyncSession = Depends(get_db),
) -> DataSourceOut:
    data_source = DataSource(
        name=payload.name,
        source_type=DataSourceType.BIGQUERY,
        external_uri=payload.external_uri,
    )
    db.add(data_source)
    await db.flush()
    await db.refresh(data_source)

    logger.info("BigQuery DataSource registered", id=str(data_source.id))
    return DataSourceOut.model_validate(data_source)


@router.get(
    "/{data_source_id}",
    response_model=DataSourceOut,
    summary="Get a data source by ID",
)
async def get_data_source(
    data_source_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DataSourceOut:
    ds = await db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="DataSource not found")
    return DataSourceOut.model_validate(ds)
