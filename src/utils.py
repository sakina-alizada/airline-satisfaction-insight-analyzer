import json
import time
import functools
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timer(func):
    """Decorator that prints the elapsed time of the wrapped function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[utils] {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def print_section_header(title: str, width: int = 60):
    """Print a formatted section header to stdout."""
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """Print shape, dtypes, and missing-value summary for a DataFrame."""
    print_section_header(f"DataFrame Summary: {name}")
    print(f"  Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory    : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Target    : '{config.TARGET_COLUMN}' distribution →")
    if config.TARGET_COLUMN in df.columns:
        vc = df[config.TARGET_COLUMN].value_counts()
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            label = "Satisfied" if val == 1 else "Neutral/Dissatisfied"
            print(f"             {label}: {cnt:,} ({pct:.1f}%)")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing   : None")
    else:
        print(f"  Missing   : {len(missing)} column(s) with nulls")
        for col, n in missing.items():
            print(f"             {col}: {n:,} ({n/len(df)*100:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path) -> dict:
    """Load a JSON file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path):
    """Save *data* as pretty-printed JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_serialisable)
    print(f"[utils] JSON saved → {path}")


def _json_serialisable(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


# ---------------------------------------------------------------------------
# Output directory checks
# ---------------------------------------------------------------------------

def ensure_output_dirs():
    """Create all required output subdirectories if they don't exist."""
    for d in (config.FIGURES_DIR, config.REPORTS_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("[utils] Output directories verified.")


# ---------------------------------------------------------------------------
# Dataset download hint
# ---------------------------------------------------------------------------

def check_dataset_exists() -> bool:
    """
    Return True if the dataset CSV is present.
    If not, print a helpful message about where to obtain it.
    """
    exists = config.DATASET_PATH.exists()
    if not exists:
        print(
            f"\n[utils] Dataset not found at: {config.DATASET_PATH}\n"
            "  To download the dataset:\n"
            "  1. Visit https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction\n"
            "  2. Download the CSV and rename it to:\n"
            f"     {config.DATASET_FILENAME}\n"
            f"  3. Place it in the data/ folder:\n"
            f"     {config.DATA_DIR}/\n"
        )
    return exists


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def print_run_summary(outputs: dict):
    """
    Print a final summary of all output file locations after
    the main pipeline completes.
    """
    print_section_header("Pipeline Complete — Output Locations")
    for label, path in outputs.items():
        if path:
            print(f"  {label:<30}: {path}")
    print()
    print(f"  Figures directory  : {config.FIGURES_DIR}")
    print(f"  Reports directory  : {config.REPORTS_DIR}")
    print(f"  Models directory   : {config.MODELS_DIR}")
    print()
