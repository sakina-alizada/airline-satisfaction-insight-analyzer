import argparse
import sys
import time
import traceback

from src import config
from src.utils import (
    check_dataset_exists,
    ensure_output_dirs,
    print_run_summary,
    print_section_header,
    print_dataframe_summary,
)
from src.data_processing import process_data, get_dataset_statistics
from src.visualization import generate_all_visualizations
from src.model_training import train_all_models
from src.evaluation import evaluate_all_models
from src.genai_insights import generate_insight_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Airline Passenger Satisfaction Insight Analyzer"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the CSV dataset (default: data/airline_passenger_satisfaction.csv)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip generating visualisations (faster debug runs)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip AI insight report generation",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    start_time = time.perf_counter()

    print_section_header("Airline Passenger Satisfaction Insight Analyzer", width=65)
    print("  Starting full analysis pipeline …\n")

    # ── 0. Setup ────────────────────────────────────────────────────────────
    ensure_output_dirs()

    # ── 1. Dataset check ────────────────────────────────────────────────────
    print_section_header("Step 1: Dataset Validation")
    if not check_dataset_exists():
        print(
            "[main] ERROR: Dataset file not found. Please download the dataset "
            "and place it in the data/ directory.\n"
            "See README.md → 'Dataset' section for instructions."
        )
        sys.exit(1)
    print("[main] Dataset found.")

    # ── 2. Data processing ──────────────────────────────────────────────────
    print_section_header("Step 2: Data Loading & Cleaning")
    try:
        data_path = args.data if args.data else None
        df = process_data(data_path)
        print_dataframe_summary(df, name="Cleaned Dataset")
    except Exception as exc:
        print(f"[main] ERROR during data processing: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 3. Dataset statistics (used in report) ───────────────────────────────
    stats = get_dataset_statistics(df)

    # ── 4. Visualisations ───────────────────────────────────────────────────
    if not args.skip_viz:
        print_section_header("Step 3: Exploratory Data Analysis & Visualisations")
        try:
            generate_all_visualizations(df)
        except Exception as exc:
            print(f"[main] Warning: Visualisation step failed — {exc}")
            traceback.print_exc()
            # Non-fatal: continue pipeline
    else:
        print("[main] Skipping visualisations (--skip-viz).")

    # ── 5. Model training ───────────────────────────────────────────────────
    print_section_header("Step 4: Model Training")
    try:
        training_results = train_all_models(df)
    except Exception as exc:
        print(f"[main] ERROR during model training: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 6. Model evaluation ─────────────────────────────────────────────────
    print_section_header("Step 5: Model Evaluation & Comparison")
    try:
        eval_results = evaluate_all_models(training_results)
    except Exception as exc:
        print(f"[main] ERROR during evaluation: {exc}")
        traceback.print_exc()
        sys.exit(1)

    rf_metrics  = eval_results["rf_metrics"]
    lr_metrics  = eval_results["lr_metrics"]
    comparison  = eval_results["comparison"]
    rf_importance = training_results["rf_importance"]

    # ── 7. AI insight report ────────────────────────────────────────────────
    if not args.skip_report:
        print_section_header("Step 6: AI Insight Report Generation")
        try:
            report_path = generate_insight_report(
                stats         = stats,
                rf_metrics    = rf_metrics,
                lr_metrics    = lr_metrics,
                comparison    = comparison,
                importance_df = rf_importance,
            )
        except Exception as exc:
            print(f"[main] Warning: Report generation failed — {exc}")
            traceback.print_exc()
            report_path = None
    else:
        print("[main] Skipping report generation (--skip-report).")
        report_path = None

    # ── 8. Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    print_run_summary({
        "Insight Report":          report_path,
        "RF Model":                config.MODELS_DIR / config.RF_MODEL_FILENAME,
        "LR Model":                config.MODELS_DIR / config.LR_MODEL_FILENAME,
        "Model Metrics (JSON)":    config.MODELS_DIR / config.METRICS_FILENAME,
        "Feature Importance (CSV)": config.MODELS_DIR / "rf_feature_importance.csv",
    })
    print(f"  Total pipeline time  : {elapsed:.1f}s")
    print(f"  Best model           : {comparison.get('best_model', 'N/A')}")
    print(
        f"  RF  F1 / AUC         : "
        f"{rf_metrics.get('f1_score', 'N/A')} / {rf_metrics.get('roc_auc', 'N/A')}"
    )
    print(
        f"  LR  F1 / AUC         : "
        f"{lr_metrics.get('f1_score', 'N/A')} / {lr_metrics.get('roc_auc', 'N/A')}"
    )
    print()


if __name__ == "__main__":
    main()
