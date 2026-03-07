# src/model_training.py
# Builds, trains, and persists the two classification models:
#   1. RandomForestClassifier  (Model 1)
#   2. LogisticRegression      (Model 2)
#
# Each model is wrapped in a full sklearn Pipeline that includes
# the preprocessing step built in data_processing.py.

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data_processing import (
    build_preprocessor,
    detect_feature_types,
    split_data,
)


# ---------------------------------------------------------------------------
# Build model pipelines
# ---------------------------------------------------------------------------

def build_random_forest_pipeline(X_train: pd.DataFrame):
    """
    Return a sklearn Pipeline:
        preprocessor → RandomForestClassifier
    """
    num_feats, cat_feats = detect_feature_types(
        pd.concat([X_train, pd.Series([0] * len(X_train), name=config.TARGET_COLUMN)], axis=1)
    )
    preprocessor = build_preprocessor(num_feats, cat_feats)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(**config.RF_PARAMS)),
    ])
    return pipeline


def build_logistic_regression_pipeline(X_train: pd.DataFrame):
    """
    Return a sklearn Pipeline:
        preprocessor → LogisticRegression
    """
    num_feats, cat_feats = detect_feature_types(
        pd.concat([X_train, pd.Series([0] * len(X_train), name=config.TARGET_COLUMN)], axis=1)
    )
    preprocessor = build_preprocessor(num_feats, cat_feats)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(**config.LR_PARAMS)),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                model_name: str = "model") -> Pipeline:
    """Fit *pipeline* on training data and return it."""
    print(f"\n[model_training] Training {model_name} …")
    pipeline.fit(X_train, y_train)
    print(f"[model_training] {model_name} training complete.")
    return pipeline


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, filename: str):
    """Save a fitted pipeline to outputs/models/<filename>."""
    path = config.MODELS_DIR / filename
    joblib.dump(pipeline, path)
    print(f"[model_training] Model saved → {path}")
    return path


def load_model(filename: str) -> Pipeline:
    """Load a previously saved pipeline from outputs/models/<filename>."""
    path = config.MODELS_DIR / filename
    pipeline = joblib.load(path)
    print(f"[model_training] Model loaded ← {path}")
    return pipeline


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame,
                            top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importances from a RandomForestClassifier pipeline.
    Returns a DataFrame sorted by importance descending.
    """
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        print("[model_training] Classifier does not expose feature_importances_.")
        return pd.DataFrame()

    # Recover feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return importance_df


# ---------------------------------------------------------------------------
# Master training function — called from main.py
# ---------------------------------------------------------------------------

def train_all_models(df: pd.DataFrame):
    """
    Full training workflow:
      1. Split data
      2. Train Random Forest
      3. Train Logistic Regression
      4. Save both models
      5. Extract feature importances

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        rf_pipeline, lr_pipeline,
        rf_importance
    """
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Random Forest -------------------------------------------------------
    rf_pipeline = build_random_forest_pipeline(X_train)
    rf_pipeline = train_model(rf_pipeline, X_train, y_train,
                              model_name="RandomForestClassifier")
    save_model(rf_pipeline, config.RF_MODEL_FILENAME)

    # --- Logistic Regression -------------------------------------------------
    lr_pipeline = build_logistic_regression_pipeline(X_train)
    lr_pipeline = train_model(lr_pipeline, X_train, y_train,
                              model_name="LogisticRegression")
    save_model(lr_pipeline, config.LR_MODEL_FILENAME)

    # --- Feature importance (RF only) ----------------------------------------
    rf_importance = get_feature_importance(rf_pipeline, X_train)
    if not rf_importance.empty:
        imp_path = config.MODELS_DIR / "rf_feature_importance.csv"
        rf_importance.to_csv(imp_path, index=False)
        print(f"[model_training] Feature importance saved → {imp_path}")

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "rf_pipeline":   rf_pipeline,
        "lr_pipeline":   lr_pipeline,
        "rf_importance": rf_importance,
    }
