from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import polars as pl
import structlog

logger = structlog.get_logger()


class DataSourceConnector(Protocol):
    """Protocol every connector must implement."""

    async def load(self) -> pl.DataFrame:
        """Load data and return a Polars DataFrame."""
        ...


# ── File Connectors ───────────────────────────────────────────────────────────

class CSVConnector:
    """Load a CSV file from local path or MinIO-downloaded bytes."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    async def load(self) -> pl.DataFrame:
        logger.info("Loading CSV", path=str(self.file_path))
        df = pl.read_csv(self.file_path, infer_schema_length=10_000)
        logger.info("CSV loaded", rows=len(df), columns=len(df.columns))
        return df


class ExcelConnector:
    """Load an Excel (.xlsx) file."""

    def __init__(self, file_path: str | Path, sheet_name: str | int = 0) -> None:
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name

    async def load(self) -> pl.DataFrame:
        logger.info("Loading Excel", path=str(self.file_path), sheet=self.sheet_name)
        df = pl.read_excel(self.file_path, sheet_name=self.sheet_name)
        logger.info("Excel loaded", rows=len(df), columns=len(df.columns))
        return df


# ── Google Sheets Connector ───────────────────────────────────────────────────

class GoogleSheetsConnector:
    """Pull a Google Sheets tab into a Polars DataFrame via gspread."""

    def __init__(
        self,
        spreadsheet_url: str,
        worksheet_name: str | None = None,
        service_account_json: str | None = None,
    ) -> None:
        self.spreadsheet_url = spreadsheet_url
        self.worksheet_name = worksheet_name
        self.service_account_json = service_account_json

    async def load(self) -> pl.DataFrame:
        import gspread

        logger.info("Connecting to Google Sheets", url=self.spreadsheet_url)

        if self.service_account_json:
            gc = gspread.service_account(filename=self.service_account_json)
        else:
            gc = gspread.oauth()

        spreadsheet = gc.open_by_url(self.spreadsheet_url)
        worksheet = (
            spreadsheet.worksheet(self.worksheet_name)
            if self.worksheet_name
            else spreadsheet.sheet1
        )

        records = worksheet.get_all_records()
        df = pl.DataFrame(records)
        logger.info("Google Sheets loaded", rows=len(df), columns=len(df.columns))
        return df


# ── BigQuery Connector ────────────────────────────────────────────────────────

class BigQueryConnector:
    """Execute a BigQuery query or table read and return a Polars DataFrame."""

    def __init__(
        self,
        project_id: str,
        query: str | None = None,
        table_ref: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        if not query and not table_ref:
            raise ValueError("Provide either `query` or `table_ref`.")
        self.project_id = project_id
        self.query = query or f"SELECT * FROM `{table_ref}`"
        self.credentials_path = credentials_path

    async def load(self) -> pl.DataFrame:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        logger.info("Querying BigQuery", project=self.project_id)

        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            client = bigquery.Client(project=self.project_id, credentials=credentials)
        else:
            client = bigquery.Client(project=self.project_id)

        query_job = client.query(self.query)
        rows = query_job.result()
        data = [dict(row) for row in rows]
        df = pl.DataFrame(data)
        logger.info("BigQuery loaded", rows=len(df), columns=len(df.columns))
        return df


# ── Factory ───────────────────────────────────────────────────────────────────

class ConnectorFactory:
    """Build the right connector from a DataSource record."""

    @staticmethod
    def from_source_type(source_type: str, **kwargs: object) -> DataSourceConnector:
        mapping: dict[str, type] = {
            "csv": CSVConnector,
            "excel": ExcelConnector,
            "google_sheets": GoogleSheetsConnector,
            "bigquery": BigQueryConnector,
        }
        cls = mapping.get(source_type)
        if cls is None:
            raise ValueError(f"Unknown source type: {source_type!r}")
        return cls(**kwargs)  # type: ignore[return-value]
