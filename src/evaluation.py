# src/evaluation.py
# Computes evaluation metrics, plots confusion matrices and feature
# importance charts, compares both models, and persists results to disk.

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "Model") -> dict:
    """
    Compute classification metrics and return them as a dictionary.

    Parameters
    ----------
    y_true   : array-like of true binary labels
    y_pred   : array-like of predicted binary labels
    y_prob   : array-like of predicted probabilities for class 1 (optional)
    model_name : display name for log messages
    """
    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=["Neutral/Dissatisfied", "Satisfied"],
        zero_division=0,
    )
    print(f"\n[evaluation] ── {model_name} ──────────────────────────")
    for k, v in metrics.items():
        if k not in ("classification_report", "model"):
            print(f"  {k:12s}: {v}")
    print(metrics["classification_report"])
    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str):
    """Save a heatmap confusion matrix to outputs/figures/."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Neutral/\nDissatisfied", "Satisfied"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    fig.tight_layout()

    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Confusion matrix saved → {path}")
    return path


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curves(y_test, rf_prob, lr_prob):
    """Plot ROC curves for both models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for prob, label, color in [
        (rf_prob, "Random Forest", "#2ECC71"),
        (lr_prob, "Logistic Regression", "#3498DB"),
    ]:
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()

    path = config.FIGURES_DIR / "09_roc_curves.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] ROC curves saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Horizontal bar chart of Random Forest feature importances."""
    if importance_df is None or importance_df.empty:
        print("[evaluation] No feature importance data – skipping plot.")
        return None

    plot_df = importance_df.head(top_n).sort_values("importance")
    # Shorten feature names for display
    plot_df = plot_df.copy()
    plot_df["feature"] = (
        plot_df["feature"]
        .str.replace(r"^num__|^cat__", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(plot_df["feature"], plot_df["importance"],
                   color="#2ECC71", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    fig.tight_layout()

    path = config.FIGURES_DIR / "10_feature_importance.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Feature importance plot saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(rf_metrics: dict, lr_metrics: dict) -> dict:
    """
    Print a side-by-side comparison table and return the name of
    the best-performing model based on F1 score.
    """
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    print("\n[evaluation] ═══════════════════════════════════════════════")
    print("[evaluation] Model Comparison Summary")
    print("[evaluation] ═══════════════════════════════════════════════")
    header = f"{'Metric':<15} {'Random Forest':>18} {'Logistic Regression':>22}"
    print(header)
    print("─" * len(header))
    for m in metrics_to_compare:
        rf_val = rf_metrics.get(m, "N/A")
        lr_val = lr_metrics.get(m, "N/A")
        rf_str = f"{rf_val:.4f}" if isinstance(rf_val, float) else str(rf_val)
        lr_str = f"{lr_val:.4f}" if isinstance(lr_val, float) else str(lr_val)
        print(f"{m:<15} {rf_str:>18} {lr_str:>22}")

    rf_f1 = rf_metrics.get("f1_score", 0)
    lr_f1 = lr_metrics.get("f1_score", 0)
    best  = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
    print(f"\n[evaluation] Best model (by F1): {best}")
    print("[evaluation] ═══════════════════════════════════════════════\n")
    return {"best_model": best, "rf_f1": rf_f1, "lr_f1": lr_f1}


# ---------------------------------------------------------------------------
# Persist metrics
# ---------------------------------------------------------------------------

def save_metrics(rf_metrics: dict, lr_metrics: dict, comparison: dict):
    """Serialise metrics to outputs/models/model_metrics.json."""
    # Remove non-serialisable classification_report string
    def _clean(d):
        return {k: v for k, v in d.items() if k != "classification_report"}

    payload = {
        "random_forest":        _clean(rf_metrics),
        "logistic_regression":  _clean(lr_metrics),
        "comparison":           comparison,
    }
    path = config.MODELS_DIR / config.METRICS_FILENAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[evaluation] Metrics saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Master evaluation function — called from main.py
# ---------------------------------------------------------------------------

def evaluate_all_models(training_results: dict) -> dict:
    """
    Run the full evaluation workflow on the two trained pipelines.

    Parameters
    ----------
    training_results : dict returned by model_training.train_all_models()

    Returns
    -------
    dict with rf_metrics, lr_metrics, comparison
    """
    rf_pipeline  = training_results["rf_pipeline"]
    lr_pipeline  = training_results["lr_pipeline"]
    X_test       = training_results["X_test"]
    y_test       = training_results["y_test"]
    rf_importance = training_results["rf_importance"]

    # Predictions
    rf_pred = rf_pipeline.predict(X_test)
    lr_pred = lr_pipeline.predict(X_test)

    # Probabilities (for ROC)
    try:
        rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        rf_prob = None
    try:
        lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        lr_prob = None

    # Metrics
    rf_metrics = compute_metrics(y_test, rf_pred, rf_prob, "Random Forest")
    lr_metrics = compute_metrics(y_test, lr_pred, lr_prob, "Logistic Regression")

    # Confusion matrices
    plot_confusion_matrix(y_test, rf_pred, "Random Forest",   "11_cm_random_forest.png")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "12_cm_logistic_regression.png")

    # ROC curves
    plot_roc_curves(y_test, rf_prob, lr_prob)

    # Feature importance
    plot_feature_importance(rf_importance)

    # Comparison
    comparison = compare_models(rf_metrics, lr_metrics)

    # Persist
    save_metrics(rf_metrics, lr_metrics, comparison)

    return {
        "rf_metrics":  rf_metrics,
        "lr_metrics":  lr_metrics,
        "comparison":  comparison,
    }
