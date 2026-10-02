# AthenaMining — Backend Implementation Plan

> Full engineering roadmap for the AthenaMining AutoML platform backend.
> Architecture: FastAPI + Celery + Polars + LightGBM + MLflow + MinIO + PostgreSQL + Evidently AI.
> All code must follow: full typing, ORM-first, Clean Architecture, SOLID, document all, commit every change.

---

## Table of Contents

1. [Current State](#current-state)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1 — Data Ingestion](#phase-1--data-ingestion-status-done)
4. [Phase 2 — Data Preparation](#phase-2--data-preparation)
5. [Phase 3 — Feature Engineering](#phase-3--feature-engineering)
6. [Phase 4 — Training Session](#phase-4--training-session)
7. [Phase 5 — Predict & Data Drift](#phase-5--predict--data-drift)
8. [Phase 6 — Database Migrations](#phase-6--database-migrations)
9. [Phase 7 — Auth & Security](#phase-7--auth--security)
10. [Phase 8 — Observability & Logging](#phase-8--observability--logging)
11. [Full File Tree](#full-file-tree)
12. [API Contract Reference](#api-contract-reference)
13. [Data Flow Diagram](#data-flow-diagram)

---

## Current State

| Layer | Status | Files |
|-------|--------|-------|
| Project config | ✅ Done | `pyproject.toml`, `.env.example` |
| Infrastructure | ✅ Done | `docker-compose.yml`, `docker/Dockerfile.api` |
| Core config | ✅ Done | `app/core/config.py`, `app/core/celery_app.py` |
| DB models | ✅ Done | `app/db/models/pipeline.py`, `app/db/session.py` |
| API skeleton | ✅ Done | `app/main.py`, `app/api/v1/router.py`, endpoints scaffold |
| Ingestion connectors | ✅ Done | `app/services/ingestion/connectors.py` |
| MinIO storage | ✅ Done | `app/services/ingestion/storage.py` |
| Celery task shell | ✅ Done | `app/tasks/pipeline_tasks.py` |
| BDD specs (Phase 1) | ✅ Done | `tests/features/data_ingestion.feature` |
| data_prep | 🔴 Todo | — |
| feature_engineering | 🔴 Todo | — |
| training_session | 🔴 Todo | — |
| predict + drift | 🔴 Todo | — |
| Alembic migrations | 🔴 Todo | — |
| Auth / JWT | 🔴 Todo | — |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI (API Layer)                      │
│  /data-sources  /pipelines  /predict  /auth  /health        │
└────────────────────────────┬────────────────────────────────┘
                             │  async HTTP
              ┌──────────────▼──────────────┐
              │       Service Layer          │
              │  IngestionService            │
              │  PipelineOrchestrator        │
              │  PredictService              │
              │  DriftService                │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │              Celery Workers              │
         │  [ingestion queue]   [pipeline queue]    │
         └───┬─────────────────────────────────┬───┘
             │                                 │
   ┌─────────▼──────────┐          ┌──────────▼──────────┐
   │   DataPrepService   │          │  TrainingService     │
   │   (Polars/DuckDB)   │          │  (LightGBM/Optuna)  │
   └─────────┬──────────┘          └──────────┬──────────┘
             │                                 │
   ┌─────────▼──────────┐          ┌──────────▼──────────┐
   │FeatureEngineeringSvc│          │  MLflow Registry     │
   │(sklearn/encoders)   │          │  + MinIO artifacts   │
   └────────────────────┘          └─────────────────────┘
```

---

## Phase 1 — Data Ingestion ✅ DONE

### What was built
- `CSVConnector`, `ExcelConnector`, `GoogleSheetsConnector`, `BigQueryConnector`
- `StorageService` (MinIO upload/download/presigned URLs)
- `POST /api/v1/data-sources/upload` — file upload
- `POST /api/v1/data-sources/connect/google-sheets`
- `POST /api/v1/data-sources/connect/bigquery`
- `GET /api/v1/data-sources/{id}`
- `DataSource` and `Pipeline` ORM models
- BDD Gherkin feature specs + pytest-bdd step definitions

---

## Phase 2 — Data Preparation

> **Goal:** Download raw data from MinIO, infer schema, handle nulls and outliers, persist a clean `.parquet` back to MinIO.

### Files to create

```
app/
├── services/
│   └── pipeline/
│       ├── data_prep.py          ← main service
│       └── schema_inferrer.py    ← column type detection
tests/
├── features/
│   ├── data_prep.feature         ← BDD spec
│   └── steps/
│       └── test_data_prep.py
├── unit/
│   └── test_schema_inferrer.py
```

### `schema_inferrer.py`

```python
class ColumnType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL_LOW = "categorical_low"   # cardinality < threshold
    CATEGORICAL_HIGH = "categorical_high" # cardinality >= threshold
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"                         # free text, drop candidate

@dataclass
class ColumnSchema:
    name: str
    dtype: ColumnType
    null_pct: float
    cardinality: int
    is_target: bool

class SchemaInferrer:
    """Scan a Polars DataFrame and return a list[ColumnSchema]."""

    HIGH_CARDINALITY_THRESHOLD: ClassVar[int] = 50

    def infer(self, df: pl.DataFrame, target_column: str) -> list[ColumnSchema]: ...
```

### `data_prep.py`

```python
@dataclass
class DataPrepConfig:
    numeric_impute_strategy: Literal["mean", "median", "drop"] = "median"
    categorical_impute_value: str = "missing"
    outlier_detection: bool = True
    outlier_contamination: float = 0.05

class DataPrepService:
    """
    Responsibilities:
    1. Load raw file from MinIO via StorageService
    2. Parse with Polars (CSV/Excel) or DuckDB (large files / SQL)
    3. Run SchemaInferrer to classify columns
    4. Impute nulls:
       - numeric  → SimpleImputer(strategy=config.numeric_impute_strategy)
       - category → fill with config.categorical_impute_value
    5. Detect outliers via IsolationForest on numeric columns
       → add boolean column `__outlier_flag__`
    6. Save cleaned DataFrame as .parquet to MinIO
    7. Update DataSource.schema_json and DataSource.row_count in DB
    8. Return DataPrepResult
    """

    def __init__(
        self,
        storage: StorageService,
        db_session: AsyncSession,
        config: DataPrepConfig | None = None,
    ) -> None: ...

    async def run(
        self,
        data_source: DataSource,
        target_column: str,
    ) -> DataPrepResult: ...

@dataclass
class DataPrepResult:
    clean_parquet_path: str      # MinIO path
    schema: list[ColumnSchema]
    row_count: int
    outlier_count: int
    null_report: dict[str, float]  # col → null %
```

### BDD Scenarios (data_prep.feature)

```gherkin
Feature: Data Preparation
  Scenario: Infer schema from a mixed-type CSV
  Scenario: Impute numeric nulls with median strategy
  Scenario: Impute categorical nulls with "missing" constant
  Scenario: Flag outliers using IsolationForest
  Scenario: Save cleaned data as parquet to MinIO
  Scenario: Reject a dataset where target column does not exist
  Scenario: Update DataSource schema_json and row_count in database
```

### Commit sequence
```
feat(pipeline): add SchemaInferrer with ColumnType classification
feat(pipeline): add DataPrepService with null imputation and outlier detection
feat(pipeline): save cleaned parquet to MinIO and update DataSource in DB
test(bdd): add data_prep.feature Gherkin specs and step definitions
test(unit): add unit tests for SchemaInferrer edge cases
```

---

## Phase 3 — Feature Engineering

> **Goal:** Apply encoders, scalers, and forecasting feature extractors via a scikit-learn-compatible pipeline. Compute correlation matrix and emit Target Leakage alerts.

### Files to create

```
app/
├── services/
│   └── pipeline/
│       ├── feature_engineering.py   ← main service
│       ├── encoders.py               ← custom encoder wrappers
│       └── correlation.py            ← correlation + leakage check
tests/
├── features/
│   ├── feature_engineering.feature
│   └── steps/
│       └── test_feature_engineering.py
├── unit/
│   └── test_correlation.py
```

### `feature_engineering.py`

```python
@dataclass
class FeatureEngineeringConfig:
    scaler: Literal["standard", "robust", "none"] = "robust"
    high_card_encoder: Literal["target", "binary"] = "target"
    low_card_encoder: Literal["onehot", "ordinal"] = "onehot"
    leakage_threshold: float = 0.95  # correlation > this → alert
    # Forecasting only
    lag_periods: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 30])

class FeatureEngineeringService:
    """
    Builds a sklearn ColumnTransformer pipeline:
    - Numeric pipe:   [Imputer → Scaler]
    - Low-card pipe:  [Imputer → OneHotEncoder]
    - High-card pipe: [Imputer → TargetEncoder (with smoothing)]
    - Datetime pipe:  [DatetimeFeatureExtractor]
    - Forecasting:    [LagExtractor → RollingMeanExtractor]

    Also computes:
    - Spearman correlation matrix between all features and target
    - Target Leakage alert if any feature corr > leakage_threshold
    - Feature schema for frontend rendering
    """

    async def run(
        self,
        clean_parquet_path: str,
        schema: list[ColumnSchema],
        target_column: str,
        problem_type: ProblemType,
        config: FeatureEngineeringConfig | None = None,
    ) -> FeatureEngineeringResult: ...

@dataclass
class FeatureEngineeringResult:
    processed_parquet_path: str        # MinIO path of transformed data
    pipeline_artifact_path: str        # MinIO path of serialized sklearn Pipeline
    correlation_matrix: dict           # JSON-serializable for frontend
    leakage_alerts: list[str]          # column names that exceed threshold
    feature_names_out: list[str]
```

### `correlation.py`

```python
class CorrelationService:
    """Compute Spearman/Pearson correlation matrix using Polars."""

    def compute(
        self,
        df: pl.DataFrame,
        target_column: str,
        method: Literal["spearman", "pearson"] = "spearman",
    ) -> dict[str, dict[str, float]]: ...

    def check_target_leakage(
        self,
        correlation_matrix: dict[str, dict[str, float]],
        target_column: str,
        threshold: float = 0.95,
    ) -> list[str]: ...
```

### BDD Scenarios (feature_engineering.feature)

```gherkin
Feature: Feature Engineering
  Scenario: Apply OneHotEncoder to low-cardinality categorical columns
  Scenario: Apply TargetEncoder to high-cardinality categorical columns
  Scenario: Scale numeric columns with RobustScaler
  Scenario: Extract lag and rolling features for forecasting problem type
  Scenario: Compute Spearman correlation matrix for all features
  Scenario: Emit target leakage alert when feature correlation exceeds 0.95
  Scenario: Serialize sklearn Pipeline to joblib and upload to MinIO
```

### Commit sequence
```
feat(pipeline): add ColumnTransformer pipeline with numeric, categorical and datetime pipes
feat(pipeline): add TargetEncoder wrapper with smoothing for high-cardinality columns
feat(pipeline): add LagExtractor and RollingMeanExtractor for forecasting
feat(pipeline): add CorrelationService with Spearman matrix and target leakage check
feat(pipeline): serialize sklearn Pipeline to MinIO
test(bdd): add feature_engineering.feature Gherkin specs
test(unit): add unit tests for CorrelationService leakage detection
```

---

## Phase 4 — Training Session

> **Goal:** Competitive multi-model training with Optuna hyperparameter search. Register every trial in MLflow. Persist winning model to MinIO.

### Files to create

```
app/
├── services/
│   └── pipeline/
│       ├── training_session.py     ← orchestrates competition
│       ├── models/
│       │   ├── base_model.py       ← abstract model interface
│       │   ├── lgbm_model.py       ← LightGBM wrapper
│       │   ├── xgboost_model.py    ← XGBoost wrapper
│       │   └── nixtla_model.py     ← StatsForecast wrapper (forecasting)
│       └── hyperparameter_search.py ← Optuna study
├── services/
│   └── registry/
│       └── mlflow_registry.py      ← MLflow experiment/run management
tests/
├── features/
│   ├── training_session.feature
│   └── steps/
│       └── test_training_session.py
├── unit/
│   └── test_hyperparameter_search.py
```

### `base_model.py`

```python
class BaseMLModel(ABC):
    """Abstract interface every model adapter must implement."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> "BaseMLModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        problem_type: ProblemType,
    ) -> dict[str, float]: ...  # {"accuracy": 0.92, "f1": 0.89, ...}

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]: ...
```

### `training_session.py`

```python
@dataclass
class TrainingConfig:
    n_trials: int = 50               # Optuna trials per model
    cv_folds: int = 5
    primary_metric: str = "f1"       # metric to optimise
    time_budget_seconds: int = 600   # max training wall time
    test_size: float = 0.2
    random_state: int = 42

class TrainingSessionService:
    """
    Workflow:
    1. Load processed parquet from MinIO
    2. Split: StratifiedKFold (classification) or TimeSeriesSplit (forecasting)
    3. For each model adapter (LightGBM, XGBoost, [Nixtla for forecasting]):
       a. Create Optuna Study
       b. Run n_trials, log each trial to MLflow
       c. Store best params and CV score
    4. Select overall winner by primary_metric
    5. Retrain winner on full train set
    6. Compute final metrics on held-out test set
    7. Compute feature importance
    8. Save model artifact to MinIO via MLflow
    9. Update Pipeline record: status=COMPLETED, mlflow_run_id, model_minio_path, metrics_json
    """

    async def run(
        self,
        pipeline: Pipeline,
        processed_parquet_path: str,
        pipeline_artifact_path: str,
        schema: list[ColumnSchema],
        config: TrainingConfig | None = None,
    ) -> TrainingResult: ...

@dataclass
class TrainingResult:
    winning_model: str              # "lightgbm" | "xgboost" | "nixtla"
    best_params: dict[str, Any]
    metrics: dict[str, float]       # final test set metrics
    feature_importance: dict[str, float]
    mlflow_run_id: str
    model_minio_path: str
```

### `mlflow_registry.py`

```python
class MLflowRegistry:
    """Thin wrapper over MLflow client for AthenaMining conventions."""

    def start_run(self, pipeline_id: str, model_name: str) -> str: ...
    def log_params(self, run_id: str, params: dict[str, Any]) -> None: ...
    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, run_id: str, local_path: str) -> str: ...
    def register_model(self, run_id: str, model_name: str) -> str: ...
    def get_best_run(self, experiment_name: str, metric: str) -> str: ...
```

### `hyperparameter_search.py`

```python
class HyperparameterSearch:
    """Optuna study factory for each model type."""

    LGBM_SEARCH_SPACE: ClassVar[dict[str, Any]] = {
        "num_leaves": (20, 300),
        "learning_rate": (1e-4, 0.3),
        "n_estimators": (100, 1000),
        "min_child_samples": (5, 100),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
    }

    def create_study(
        self,
        model_cls: type[BaseMLModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: BaseCrossValidator,
        problem_type: ProblemType,
        n_trials: int,
        mlflow_run_id: str,
    ) -> optuna.Study: ...
```

### BDD Scenarios (training_session.feature)

```gherkin
Feature: Training Session
  Scenario: Train LightGBM model for classification and log to MLflow
  Scenario: Train XGBoost model for classification and log to MLflow
  Scenario: Use StratifiedKFold for classification problem type
  Scenario: Use TimeSeriesSplit for forecasting problem type
  Scenario: Select winning model by highest F1 score
  Scenario: Save winning model artifact to MinIO
  Scenario: Update Pipeline status to COMPLETED with metrics_json
  Scenario: Update Pipeline status to FAILED on training error
  Scenario: Return feature importance for all numeric and encoded features
```

### Commit sequence
```
feat(training): add BaseMLModel abstract interface
feat(training): add LightGBM and XGBoost model adapters with evaluate and feature_importance
feat(training): add Nixtla StatsForecast adapter for forecasting problem type
feat(training): add HyperparameterSearch with Optuna study and search spaces
feat(training): add MLflowRegistry wrapper for experiment tracking
feat(training): add TrainingSessionService with StratifiedKFold and TimeSeriesSplit
feat(training): persist winning model to MinIO and update Pipeline DB record
test(bdd): add training_session.feature Gherkin specs
test(unit): add unit tests for HyperparameterSearch param sampling
```

---

## Phase 5 — Predict & Data Drift

> **Goal:** Load a trained model by pipeline ID, apply the same preprocessing pipeline, generate predictions, and compute Data Drift vs. training distribution using Evidently AI.

### Files to create

```
app/
├── services/
│   └── pipeline/
│       ├── predict.py          ← inference service
│       └── drift.py            ← Evidently drift analysis
├── api/
│   └── v1/
│       └── endpoints/
│           └── predict.py      ← POST /predict, GET /drift-report
├── schemas/
│   └── predict.py              ← PredictRequest, PredictResponse, DriftReport
tests/
├── features/
│   ├── predict.feature
│   └── steps/
│       └── test_predict.py
```

### `predict.py` (service)

```python
@dataclass
class PredictRequest:
    pipeline_id: str
    records: list[dict[str, Any]]   # raw input rows

@dataclass
class PredictResponse:
    predictions: list[float | int | str]
    probabilities: list[float] | None   # for classification
    drift_alert: bool
    drift_summary: dict | None

class PredictService:
    """
    Workflow:
    1. Fetch Pipeline record from DB (must be COMPLETED)
    2. Download sklearn preprocessing Pipeline from MinIO
    3. Download trained model artifact from MinIO (via MLflow)
    4. Transform input records through the same pipeline
    5. Generate predictions (+ probabilities for classification)
    6. Async: send input data to DriftService for PSI/KL analysis
    7. Return PredictResponse
    """

    async def predict(self, request: PredictRequest) -> PredictResponse: ...
```

### `drift.py`

```python
@dataclass
class DriftReport:
    pipeline_id: str
    computed_at: datetime
    dataset_drift_detected: bool
    psi_scores: dict[str, float]         # column → PSI value
    kl_divergence: dict[str, float]      # column → KL divergence
    drifted_columns: list[str]
    severity: Literal["none", "warning", "critical"]

class DriftService:
    """
    Uses Evidently AI to compare:
    - Reference dataset: training data parquet from MinIO
    - Current dataset: incoming predict payload

    Computes:
    - Population Stability Index (PSI) per column
    - Kullback-Leibler divergence per column
    - Overall dataset drift flag

    Persists DriftReport JSON to MinIO and to DB.
    Emits severity=critical if PSI > 0.25 on target-correlated features.
    """

    async def compute(
        self,
        pipeline: Pipeline,
        current_df: pl.DataFrame,
    ) -> DriftReport: ...
```

### API Endpoints to add

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/predict/{pipeline_id}` | Batch inference + drift check |
| `GET` | `/api/v1/predict/{pipeline_id}/drift-report` | Latest drift report |
| `GET` | `/api/v1/pipelines/{id}/metrics` | Full metrics + feature importance |
| `GET` | `/api/v1/pipelines/{id}/correlation` | Correlation matrix JSON |

### BDD Scenarios (predict.feature)

```gherkin
Feature: Predict and Data Drift
  Scenario: Generate predictions for a completed classification pipeline
  Scenario: Return probabilities alongside predictions for classification
  Scenario: Apply identical preprocessing pipeline to predict input
  Scenario: Return 400 when pipeline is not in COMPLETED status
  Scenario: Compute PSI and KL divergence for incoming data
  Scenario: Set drift_alert=true when PSI exceeds 0.25 on any feature
  Scenario: Persist drift report to MinIO and database
  Scenario: Return severity=critical for high PSI on target-correlated features
```

### Commit sequence
```
feat(predict): add PredictService loading model and preprocessing pipeline from MinIO
feat(predict): add batch inference with probability output for classification
feat(drift): add DriftService using Evidently AI with PSI and KL divergence
feat(drift): add severity classification (none/warning/critical) based on PSI thresholds
feat(api): add POST /predict/{pipeline_id} and GET /drift-report endpoints
feat(db): add DriftReport ORM model and repository
test(bdd): add predict.feature Gherkin specs and step definitions
```

---

## Phase 6 — Database Migrations

> **Goal:** Set up Alembic for schema versioning. One migration per model group.

### Files to create

```
alembic/
├── env.py              ← wire to async SQLAlchemy engine
├── script.py.mako
└── versions/
    ├── 001_create_data_sources.py
    ├── 002_create_pipelines.py
    └── 003_create_drift_reports.py
```

### Commands
```bash
# Generate migration
alembic revision --autogenerate -m "create data_sources table"

# Apply
alembic upgrade head

# Rollback one
alembic downgrade -1
```

### Commit sequence
```
build(alembic): configure async Alembic env with asyncpg
feat(db/migrations): add 001_create_data_sources migration
feat(db/migrations): add 002_create_pipelines migration
feat(db/migrations): add 003_create_drift_reports migration
```

---

## Phase 7 — Auth & Security

> **Goal:** JWT-based authentication. Users own their pipelines. Permissions per resource.

### Files to create

```
app/
├── core/
│   └── security.py         ← JWT encode/decode, password hashing
├── db/
│   └── models/
│       └── user.py         ← User ORM model
├── services/
│   └── auth/
│       ├── auth_service.py ← register, login, token refresh
│       └── dependencies.py ← FastAPI get_current_user dependency
├── api/
│   └── v1/
│       └── endpoints/
│           └── auth.py     ← POST /register, POST /login, POST /refresh
├── schemas/
│   └── auth.py             ← UserCreate, UserOut, TokenResponse
tests/
└── features/
    ├── auth.feature
    └── steps/
        └── test_auth.py
```

### `user.py` model

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID]
    email: Mapped[str]          # unique index
    hashed_password: Mapped[str]
    full_name: Mapped[str]
    is_active: Mapped[bool]
    is_superuser: Mapped[bool]
    created_at: Mapped[datetime]

    pipelines: Mapped[list[Pipeline]]  # relationship
```

### Security rules
- Passwords hashed with **bcrypt** via `passlib`
- Access token: **JWT HS256**, expires in 24h
- Refresh token: **JWT HS256**, expires in 30d
- All pipeline endpoints require `get_current_active_user` dependency
- Users can only read/write their own pipelines (superuser bypasses)

### Commit sequence
```
feat(db): add User ORM model with pipeline relationship
feat(auth): add JWT encode/decode and bcrypt password hashing in security.py
feat(auth): add AuthService with register, login and refresh token flows
feat(api): add POST /auth/register, /auth/login, /auth/refresh endpoints
feat(api): protect all pipeline and data-source endpoints with JWT dependency
feat(db/migrations): add 004_create_users migration
test(bdd): add auth.feature Gherkin specs
```

---

## Phase 8 — Observability & Logging

> **Goal:** Structured logging, Celery task progress events, and pipeline status webhooks.

### Files to create

```
app/
├── core/
│   └── logging.py          ← structlog configuration
├── services/
│   └── pipeline/
│       └── events.py       ← pipeline progress event emitter
```

### Logging setup (`logging.py`)

```python
def configure_logging(environment: str) -> None:
    """
    - development: pretty colored console output
    - production: JSON lines for log aggregators (Loki, Datadog)
    Uses structlog with context vars for pipeline_id, task_id, user_id.
    """
```

### Pipeline progress events

Each Celery task step emits a progress event to Valkey pub/sub:

```python
class PipelineEvent(TypedDict):
    pipeline_id: str
    status: PipelineStatus
    step: str              # "data_prep" | "feature_engineering" | "training" | ...
    progress_pct: int      # 0-100
    message: str
    timestamp: str

class PipelineEventEmitter:
    """Publish events to Valkey channel `pipeline:{pipeline_id}:events`."""
    def emit(self, event: PipelineEvent) -> None: ...
```

### WebSocket endpoint (optional — Phase 8b)

```
GET /api/v1/pipelines/{id}/events  ← SSE stream for real-time status
```

### Commit sequence
```
feat(core): configure structlog with JSON production and pretty dev output
feat(pipeline): add PipelineEventEmitter publishing progress to Valkey pub/sub
feat(api): add SSE endpoint for real-time pipeline progress streaming
```

---

## Full File Tree

```
athena-auto-data-mining/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── auth.py              [Phase 7]
│   │       │   ├── data_sources.py      [✅ Phase 1]
│   │       │   ├── health.py            [✅ Phase 1]
│   │       │   ├── pipelines.py         [✅ Phase 1]
│   │       │   └── predict.py           [Phase 5]
│   │       └── router.py                [✅ Phase 1]
│   ├── core/
│   │   ├── celery_app.py                [✅ Phase 1]
│   │   ├── config.py                    [✅ Phase 1]
│   │   ├── logging.py                   [Phase 8]
│   │   └── security.py                  [Phase 7]
│   ├── db/
│   │   ├── models/
│   │   │   ├── pipeline.py              [✅ Phase 1]
│   │   │   ├── user.py                  [Phase 7]
│   │   │   └── drift_report.py          [Phase 5]
│   │   ├── repositories/
│   │   │   ├── pipeline_repository.py   [Phase 2]
│   │   │   ├── data_source_repository.py[Phase 2]
│   │   │   └── user_repository.py       [Phase 7]
│   │   └── session.py                   [✅ Phase 1]
│   ├── schemas/
│   │   ├── auth.py                      [Phase 7]
│   │   ├── data_source.py               [✅ Phase 1]
│   │   └── predict.py                   [Phase 5]
│   ├── services/
│   │   ├── auth/
│   │   │   ├── auth_service.py          [Phase 7]
│   │   │   └── dependencies.py          [Phase 7]
│   │   ├── ingestion/
│   │   │   ├── connectors.py            [✅ Phase 1]
│   │   │   └── storage.py               [✅ Phase 1]
│   │   └── pipeline/
│   │       ├── orchestrator.py          [Phase 2 — ties all phases]
│   │       ├── data_prep.py             [Phase 2]
│   │       ├── schema_inferrer.py       [Phase 2]
│   │       ├── feature_engineering.py   [Phase 3]
│   │       ├── encoders.py              [Phase 3]
│   │       ├── correlation.py           [Phase 3]
│   │       ├── training_session.py      [Phase 4]
│   │       ├── hyperparameter_search.py [Phase 4]
│   │       ├── models/
│   │       │   ├── base_model.py        [Phase 4]
│   │       │   ├── lgbm_model.py        [Phase 4]
│   │       │   ├── xgboost_model.py     [Phase 4]
│   │       │   └── nixtla_model.py      [Phase 4]
│   │       ├── predict.py               [Phase 5]
│   │       ├── drift.py                 [Phase 5]
│   │       └── events.py               [Phase 8]
│   ├── services/
│   │   └── registry/
│   │       └── mlflow_registry.py       [Phase 4]
│   ├── tasks/
│   │   ├── ingestion_tasks.py           [Phase 2]
│   │   └── pipeline_tasks.py            [✅ Phase 1]
│   └── main.py                          [✅ Phase 1]
├── alembic/
│   └── versions/
│       ├── 001_create_data_sources.py   [Phase 6]
│       ├── 002_create_pipelines.py      [Phase 6]
│       ├── 003_create_drift_reports.py  [Phase 6]
│       └── 004_create_users.py          [Phase 6]
├── tests/
│   ├── features/
│   │   ├── data_ingestion.feature       [✅ Phase 1]
│   │   ├── data_prep.feature            [Phase 2]
│   │   ├── feature_engineering.feature  [Phase 3]
│   │   ├── training_session.feature     [Phase 4]
│   │   ├── predict.feature              [Phase 5]
│   │   └── auth.feature                 [Phase 7]
│   ├── unit/
│   │   ├── test_schema_inferrer.py      [Phase 2]
│   │   ├── test_correlation.py          [Phase 3]
│   │   └── test_hyperparameter_search.py[Phase 4]
│   └── integration/
│       └── test_pipeline_e2e.py         [Phase 4+]
├── docker/
│   └── Dockerfile.api                   [✅ Phase 1]
├── docs/
│   └── logo.png                         [✅ Done]
├── docker-compose.yml                   [✅ Phase 1]
├── pyproject.toml                       [✅ Phase 1]
├── .env.example                         [✅ Phase 1]
└── README.md                            [✅ Done]
```

---

## API Contract Reference

### Data Sources
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/data-sources/upload` | 🔒 JWT | Upload CSV/Excel |
| `POST` | `/api/v1/data-sources/connect/google-sheets` | 🔒 JWT | Register Sheets URL |
| `POST` | `/api/v1/data-sources/connect/bigquery` | 🔒 JWT | Register BQ table |
| `GET` | `/api/v1/data-sources/{id}` | 🔒 JWT | Get data source |

### Pipelines
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/pipelines` | 🔒 JWT | Create + enqueue pipeline |
| `GET` | `/api/v1/pipelines/{id}` | 🔒 JWT | Status + result |
| `GET` | `/api/v1/pipelines/{id}/metrics` | 🔒 JWT | Final metrics + importance |
| `GET` | `/api/v1/pipelines/{id}/correlation` | 🔒 JWT | Correlation matrix |
| `GET` | `/api/v1/pipelines/{id}/events` | 🔒 JWT | SSE progress stream |

### Predict
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/predict/{pipeline_id}` | 🔒 JWT | Batch inference |
| `GET` | `/api/v1/predict/{pipeline_id}/drift-report` | 🔒 JWT | Latest drift report |

### Auth
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/auth/register` | Public | Create account |
| `POST` | `/api/v1/auth/login` | Public | Get JWT tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh token | New access token |

---

## Data Flow Diagram

```
User uploads CSV
       │
       ▼
POST /data-sources/upload
       │ saves raw file
       ▼
   MinIO (athena-raw/)
       │ event published
       ▼
 Valkey (ingestion queue)
       │
       ▼
Celery Worker [ingestion]
       │
       ├─► DataPrepService
       │     ├─ SchemaInferrer      → infer column types
       │     ├─ NullImputer         → fill nulls
       │     ├─ OutlierDetector     → IsolationForest flag
       │     └─ save clean.parquet  → MinIO (athena-raw/clean/)
       │
       ├─► FeatureEngineeringService
       │     ├─ ColumnTransformer   → encode + scale
       │     ├─ CorrelationService  → Spearman matrix + leakage check
       │     ├─ save processed.parquet → MinIO
       │     └─ save pipeline.joblib   → MinIO (athena-models/)
       │
       ├─► TrainingSessionService
       │     ├─ HyperparameterSearch  → Optuna per model
       │     ├─ MLflowRegistry        → log every trial
       │     ├─ Model competition      → LightGBM vs XGBoost
       │     ├─ Winner selection       → best F1/RMSE
       │     └─ save model artifact    → MinIO via MLflow
       │
       └─► Update Pipeline DB record
             status=COMPLETED
             mlflow_run_id
             model_minio_path
             metrics_json

─────────────────────────────────────────────────────

User submits predict request
       │
       ▼
POST /predict/{pipeline_id}
       │
       ▼
  PredictService
       ├─ load pipeline.joblib from MinIO
       ├─ load model artifact from MinIO
       ├─ transform input → same pipeline
       ├─ model.predict() + probabilities
       │
       └─► DriftService (async)
             ├─ load training parquet from MinIO
             ├─ Evidently DatasetDriftReport
             ├─ compute PSI + KL divergence
             ├─ severity classification
             └─ persist DriftReport → DB + MinIO

       │
       ▼
  PredictResponse
  { predictions, probabilities, drift_alert, drift_summary }
```

---

## Implementation Order (Priority)

```
Week 1:  Phase 2 — data_prep + schema_inferrer
Week 1:  Phase 6 — Alembic migrations (001, 002)
Week 2:  Phase 3 — feature_engineering + correlation
Week 2:  Phase 4 — training_session + model adapters
Week 3:  Phase 4 — MLflow registry + Optuna integration
Week 3:  Phase 5 — predict endpoint + drift with Evidently
Week 4:  Phase 7 — auth (JWT + User model + migration 004)
Week 4:  Phase 8 — structured logging + SSE progress events
```

> [!IMPORTANT]
> The `PipelineOrchestrator` class (`app/services/pipeline/orchestrator.py`) must be implemented **before** Phase 2 begins — it is the coordinator called by the Celery task and it wires all phases together sequentially, updating `Pipeline.status` between each step.

> [!TIP]
> Always write the BDD `.feature` spec first, then implement the service. This keeps each phase testable and self-documenting from day one.
