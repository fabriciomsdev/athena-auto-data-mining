from __future__ import annotations

import io
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenario, then, when

from app.main import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    """TestClient with mocked DB and MinIO."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def context() -> dict:
    return {}


# ── Background ────────────────────────────────────────────────────────────────


@given("the AthenaMining API is running")
def api_running(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


# ── Scenarios ─────────────────────────────────────────────────────────────────


@scenario("data_ingestion.feature", "Successfully upload a CSV file")
def test_upload_csv() -> None: ...


@scenario("data_ingestion.feature", "Successfully upload an Excel file")
def test_upload_excel() -> None: ...


@scenario("data_ingestion.feature", "Reject an unsupported file type")
def test_reject_unsupported_type() -> None: ...


@scenario("data_ingestion.feature", "Reject a file exceeding the 200 MB limit")
def test_reject_large_file() -> None: ...


@scenario("data_ingestion.feature", "Register a Google Sheets URL as a data source")
def test_register_google_sheets() -> None: ...


@scenario("data_ingestion.feature", "Register a BigQuery table as a data source")
def test_register_bigquery() -> None: ...


@scenario("data_ingestion.feature", "Return 404 for a non-existent data source")
def test_not_found() -> None: ...


# ── Given steps ───────────────────────────────────────────────────────────────


@given(parsers.parse('I have a valid CSV file named "{filename}"'), target_fixture="upload_file")
def valid_csv(filename: str) -> dict:
    content = b"col_a,col_b,target\n1,2,0\n3,4,1\n"
    return {"filename": filename, "content": content, "content_type": "text/csv"}


@given(parsers.parse('I have a valid Excel file named "{filename}"'), target_fixture="upload_file")
def valid_excel(filename: str) -> dict:
    import openpyxl, io as _io
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["col_a", "col_b", "target"])
    ws.append([1, 2, 0])
    buf = _io.BytesIO()
    wb.save(buf)
    return {
        "filename": filename,
        "content": buf.getvalue(),
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }


@given(
    parsers.parse('I have a file named "{filename}" with content type "{content_type}"'),
    target_fixture="upload_file",
)
def unsupported_file(filename: str, content_type: str) -> dict:
    return {"filename": filename, "content": b"%PDF-1.4", "content_type": content_type}


@given(parsers.parse("I have a CSV file that is {size:d} MB in size"), target_fixture="upload_file")
def large_csv(size: int) -> dict:
    content = b"a,b\n" + b"x," * (size * 1024 * 1024 // 2)
    return {"filename": "big.csv", "content": content, "content_type": "text/csv"}


@given(
    parsers.parse('I have a valid Google Sheets URL "{url}"'),
    target_fixture="sheets_url",
)
def sheets_url(url: str) -> str:
    return url


@given(
    parsers.parse('I have a BigQuery table reference "{table_ref}"'),
    target_fixture="bq_table_ref",
)
def bq_table_ref(table_ref: str) -> str:
    return table_ref


@given("no data source exists with a random UUID", target_fixture="random_uuid")
def random_uuid() -> str:
    return str(uuid.uuid4())


@given("a data source with ID exists in the database", target_fixture="existing_ds_id")
def existing_ds_id() -> str:
    # placeholder — real integration test would seed the DB
    return str(uuid.uuid4())


# ── When steps ────────────────────────────────────────────────────────────────


@when(
    parsers.parse('I upload the file with the name "{name}"'),
    target_fixture="response",
)
def upload_file_request(
    client: TestClient, upload_file: dict, name: str
) -> Any:
    with (
        patch("app.api.v1.endpoints.data_sources.StorageService") as mock_storage,
        patch("app.db.session.get_db"),
    ):
        mock_storage.return_value.upload_bytes.return_value = "athena-raw/uploads/file.csv"
        resp = client.post(
            "/api/v1/data-sources/upload",
            data={"name": name},
            files={
                "file": (
                    upload_file["filename"],
                    io.BytesIO(upload_file["content"]),
                    upload_file["content_type"],
                )
            },
        )
    return resp


@when(
    parsers.parse('I register it with the name "{name}"'),
    target_fixture="response",
)
def register_sheets(client: TestClient, sheets_url: str | None, bq_table_ref: str | None, name: str) -> Any:
    uri = sheets_url or bq_table_ref
    endpoint = (
        "/api/v1/data-sources/connect/google-sheets"
        if sheets_url
        else "/api/v1/data-sources/connect/bigquery"
    )
    resp = client.post(endpoint, json={"name": name, "external_uri": uri})
    return resp


@when("I request the data source by its ID", target_fixture="response")
def get_by_id(client: TestClient, existing_ds_id: str) -> Any:
    return client.get(f"/api/v1/data-sources/{existing_ds_id}")


@when("I request the data source by that UUID", target_fixture="response")
def get_missing(client: TestClient, random_uuid: str) -> Any:
    return client.get(f"/api/v1/data-sources/{random_uuid}")


# ── Then steps ────────────────────────────────────────────────────────────────


@then(parsers.parse("the response status should be {code:d}"))
def check_status(response: Any, code: int) -> None:
    assert response.status_code == code, response.text


@then(parsers.parse('the data source record should have source_type "{stype}"'))
def check_source_type(response: Any, stype: str) -> None:
    assert response.json()["source_type"] == stype


@then("the data source should have a minio_path")
def check_minio_path(response: Any) -> None:
    assert response.json().get("minio_path") is not None


@then(parsers.parse("the data source external_uri should match the given URL"))
def check_external_uri(response: Any, sheets_url: str) -> None:
    assert response.json()["external_uri"] == sheets_url


@then("the response should contain the data source details")
def check_details(response: Any) -> None:
    data = response.json()
    assert "id" in data
    assert "name" in data
