Feature: Data Preparation
  As the AthenaMining pipeline engine
  I want to clean and standardise raw datasets
  So that they are ready for feature engineering without data leakage

  Background:
    Given the data preparation service is configured with default settings

  # ── Schema inference ────────────────────────────────────────────────────────

  Scenario: Infer schema from a mixed-type CSV
    Given a DataFrame with columns: numeric "age", categorical "city", boolean "churned", datetime "signup_date"
    When I run schema inference with target column "churned"
    Then "age" should be classified as "numeric"
    And "city" should be classified as "categorical_low"
    And "churned" should be classified as "boolean" and is_target True
    And "signup_date" should be classified as "datetime"

  Scenario: Classify high-cardinality string column as categorical_high
    Given a DataFrame with a string column "product_id" with 200 unique values
    When I run schema inference with target column "target"
    Then "product_id" should be classified as "categorical_high"

  Scenario: Classify very high cardinality string column as text
    Given a DataFrame with a string column "comment" with 600 unique values
    When I run schema inference with target column "target"
    Then "comment" should be classified as "text"

  # ── Null imputation ─────────────────────────────────────────────────────────

  Scenario: Impute numeric nulls with median strategy
    Given a numeric column "salary" with values [10000, 20000, null, 40000]
    When I apply data preparation with numeric_impute_strategy "median"
    Then the null in "salary" should be filled with 20000.0

  Scenario: Impute numeric nulls with mean strategy
    Given a numeric column "score" with values [10.0, 20.0, null, 30.0]
    When I apply data preparation with numeric_impute_strategy "mean"
    Then the null in "score" should be filled with 20.0

  Scenario: Impute categorical nulls with constant "missing"
    Given a categorical column "department" with a null value
    When I apply data preparation with default settings
    Then the null in "department" should be filled with "missing"

  # ── Outlier detection ───────────────────────────────────────────────────────

  Scenario: Flag outliers with IsolationForest
    Given a DataFrame with numeric columns containing extreme outlier values
    When I run data preparation with outlier_detection enabled
    Then the output DataFrame should contain a boolean column "__outlier_flag__"
    And at least one row should have "__outlier_flag__" set to True

  Scenario: Add __outlier_flag__ as False when no numeric columns present
    Given a DataFrame with only categorical columns
    When I run data preparation with outlier_detection enabled
    Then "__outlier_flag__" should be all False

  # ── Text column dropping ────────────────────────────────────────────────────

  Scenario: Drop free-text columns automatically
    Given a DataFrame with a "text" type column "user_review"
    When I run data preparation with drop_text_columns enabled
    Then "user_review" should not be in the output DataFrame

  # ── Target validation ───────────────────────────────────────────────────────

  Scenario: Reject a dataset where target column does not exist
    Given a DataFrame without a column named "missing_target"
    When I run data preparation with target column "missing_target"
    Then a ValueError should be raised
