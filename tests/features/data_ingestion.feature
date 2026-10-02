Feature: Data Ingestion
  As a user of AthenaMining
  I want to upload data files or connect external sources
  So that I can use them to train AutoML models

  Background:
    Given the AthenaMining API is running

  # ── File Upload ─────────────────────────────────────────────────────────────

  Scenario: Successfully upload a CSV file
    Given I have a valid CSV file named "sales_data.csv"
    When I upload the file with the name "Q1 Sales"
    Then the response status should be 201
    And the data source record should have source_type "csv"
    And the data source should have a minio_path

  Scenario: Successfully upload an Excel file
    Given I have a valid Excel file named "inventory.xlsx"
    When I upload the file with the name "Inventory 2024"
    Then the response status should be 201
    And the data source record should have source_type "excel"

  Scenario: Reject an unsupported file type
    Given I have a file named "report.pdf" with content type "application/pdf"
    When I upload the file with the name "PDF Report"
    Then the response status should be 415

  Scenario: Reject a file exceeding the 200 MB limit
    Given I have a CSV file that is 201 MB in size
    When I upload the file with the name "Huge File"
    Then the response status should be 413

  # ── Google Sheets ───────────────────────────────────────────────────────────

  Scenario: Register a Google Sheets URL as a data source
    Given I have a valid Google Sheets URL "https://docs.google.com/spreadsheets/d/abc123"
    When I register it with the name "Leads Sheet"
    Then the response status should be 201
    And the data source record should have source_type "google_sheets"
    And the data source external_uri should match the given URL

  # ── BigQuery ────────────────────────────────────────────────────────────────

  Scenario: Register a BigQuery table as a data source
    Given I have a BigQuery table reference "my_project.dataset.table"
    When I register it with the name "Sales BigQuery Table"
    Then the response status should be 201
    And the data source record should have source_type "bigquery"

  # ── Retrieval ───────────────────────────────────────────────────────────────

  Scenario: Retrieve a data source by ID
    Given a data source with ID exists in the database
    When I request the data source by its ID
    Then the response status should be 200
    And the response should contain the data source details

  Scenario: Return 404 for a non-existent data source
    Given no data source exists with a random UUID
    When I request the data source by that UUID
    Then the response status should be 404
