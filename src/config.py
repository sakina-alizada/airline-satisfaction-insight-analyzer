import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (one level up from this file)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR  = OUTPUT_DIR / "models"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_FILENAME = "airline_passenger_satisfaction.csv"
DATASET_PATH     = DATA_DIR / DATASET_FILENAME

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
TARGET_COLUMN     = "satisfaction"
# Raw values that map to the POSITIVE class (satisfied)
POSITIVE_LABELS   = {"satisfied"}
# Binary mapping applied to the target column
SATISFACTION_MAP  = {
    "satisfied":             1,
    "neutral or dissatisfied": 0,
    # some dataset versions use slightly different wording
    "dissatisfied":          0,
    "neutral":               0,
}

# ---------------------------------------------------------------------------
# Columns that must be dropped before modelling
# (identifiers / leakage columns)
# ---------------------------------------------------------------------------
DROP_COLUMNS = ["id", "unnamed: 0"]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth":    15,
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

LR_PARAMS = {
    "max_iter":     1000,
    "random_state": RANDOM_STATE,
    "solver":       "lbfgs",
}

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
RF_MODEL_FILENAME   = "random_forest_model.joblib"
LR_MODEL_FILENAME   = "logistic_regression_model.joblib"
REPORT_FILENAME     = "ai_insight_report.md"
METRICS_FILENAME    = "model_metrics.json"

# ---------------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------------
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"

# Seaborn / matplotlib palette
PALETTE_BINARY = ["#E74C3C", "#2ECC71"]   # red=dissatisfied, green=satisfied
PALETTE_MAIN   = "Set2"

# ---------------------------------------------------------------------------
# GenAI / OpenAI settings
# ---------------------------------------------------------------------------
OPENAI_MODEL      = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 2000

# ---------------------------------------------------------------------------
# Ensure output directories exist at import time
# ---------------------------------------------------------------------------
for _d in (FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
