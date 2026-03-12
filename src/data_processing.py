import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_dataset(path=None):
    """
    Load the CSV dataset from *path* (defaults to config.DATASET_PATH).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names.
    """
    path = path or config.DATASET_PATH
    print(f"[data_processing] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[data_processing] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Column-name normalisation
# ---------------------------------------------------------------------------

def _to_snake_case(name: str) -> str:
    """Convert an arbitrary column name to lowercase snake_case."""
    name = name.strip()
    # Replace common separators with underscore
    name = re.sub(r"[\s\-]+", "_", name)
    # Insert underscore between camelCase transitions
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Collapse multiple underscores and lower-case everything
    name = re.sub(r"_+", "_", name).lower()
    return name


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with all column names converted to snake_case."""
    df = df.copy()
    df.columns = [_to_snake_case(c) for c in df.columns]
    print(f"[data_processing] Columns after standardisation: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first candidate column name that actually exists in *df*.
    Returns None if none are found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a cleaned copy:
      - Drop identifier / leakage columns
      - Remove duplicate rows
      - Handle missing values (numeric → median, categorical → mode)
      - Encode the target column as binary int
    """
    df = df.copy()

    # --- Drop unwanted columns (IDs, unnamed index) -------------------------
    drop_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"[data_processing] Dropped columns: {drop_cols}")

    # --- Drop exact duplicates ----------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[data_processing] Removed {before - len(df)} duplicate rows.")

    # --- Resolve target column (may be named slightly differently) ----------
    target_col = _resolve_column(df, [config.TARGET_COLUMN, "satisfaction_v2"])
    if target_col is None:
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    if target_col != config.TARGET_COLUMN:
        df.rename(columns={target_col: config.TARGET_COLUMN}, inplace=True)

    # --- Encode target as binary int ----------------------------------------
    df[config.TARGET_COLUMN] = (
        df[config.TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(config.SATISFACTION_MAP)
    )
    n_null_target = df[config.TARGET_COLUMN].isna().sum()
    if n_null_target > 0:
        print(f"[data_processing] Warning: {n_null_target} unmapped target values → dropped.")
        df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)

    # --- Handle missing values -----------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric: fill with column median
    for c in num_cols:
        if df[c].isna().any():
            median = df[c].median()
            df[c].fillna(median, inplace=True)

    # Categorical: fill with column mode
    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode()[0]
            df[c].fillna(mode_val, inplace=True)

    missing_after = df.isna().sum().sum()
    print(f"[data_processing] Missing values after cleaning: {missing_after}")
    print(f"[data_processing] Final dataset shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Feature type detection
# ---------------------------------------------------------------------------

def detect_feature_types(df: pd.DataFrame, target: str = config.TARGET_COLUMN):
    """
    Return (numeric_features, categorical_features) lists,
    excluding the target column.
    """
    feature_df = df.drop(columns=[target], errors="ignore")
    numeric_features     = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[data_processing] Numeric features   ({len(numeric_features)}): {numeric_features}")
    print(f"[data_processing] Categorical features ({len(categorical_features)}): {categorical_features}")
    return numeric_features, categorical_features


# ---------------------------------------------------------------------------
# 5. Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_features: list, categorical_features: list):
    """
    Build and return a ColumnTransformer that:
      - Imputes + scales numeric features
      - Imputes + one-hot encodes categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 6. Train / test split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, target: str = config.TARGET_COLUMN):
    """
    Split *df* into X_train, X_test, y_train, y_test using
    the ratio and seed defined in config.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.TEST_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y,
    )
    print(
        f"[data_processing] Train: {len(X_train):,} rows | "
        f"Test: {len(X_test):,} rows"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 7. Full preprocessing entry-point (used by main.py)
# ---------------------------------------------------------------------------

def process_data(path=None):
    """
    Convenience wrapper: load → standardise → clean → return cleaned df.
    """
    df = load_dataset(path)
    df = standardize_column_names(df)
    df = clean_data(df)
    return df


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of descriptive statistics for the insight report.
    """
    target = config.TARGET_COLUMN
    stats = {
        "total_records":      len(df),
        "num_features":       df.shape[1] - 1,
        "satisfaction_rate":  round(df[target].mean() * 100, 2),
        "dissatisfaction_rate": round((1 - df[target].mean()) * 100, 2),
        "numeric_stats":      df.describe().to_dict(),
    }

    # Value-count breakdowns for key categorical columns
    for col in ["customer_type", "type_of_travel", "class"]:
        if col in df.columns:
            vc = df.groupby(col)[target].mean().round(4).to_dict()
            stats[f"{col}_satisfaction_rate"] = vc

    return stats
