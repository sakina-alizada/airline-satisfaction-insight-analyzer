# src/genai_insights.py
# Generates a natural-language insight report from model metrics,
# dataset statistics, and feature importances.
#
# Strategy
# --------
# 1. If an OPENAI_API_KEY is set, call the OpenAI Chat API to produce
#    a polished AI-generated report.
# 2. Otherwise, fall back to a rule-based template that uses the same
#    data to produce a detailed, professional Markdown document.

import os
import json
import textwrap
from datetime import date
from pathlib import Path

from src import config


# ---------------------------------------------------------------------------
# OpenAI integration (optional)
# ---------------------------------------------------------------------------

def _call_openai_api(prompt: str) -> str:
    """
    Call the OpenAI Chat Completions API and return the response text.
    Raises ImportError if `openai` is not installed.
    Raises EnvironmentError if OPENAI_API_KEY is not set.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("The `openai` package is not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=config.OPENAI_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior data scientist writing an executive insight report "
                    "for airline management. Use clear, professional language. "
                    "Format the report in GitHub Markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _build_openai_prompt(stats: dict, rf_metrics: dict, lr_metrics: dict,
                          comparison: dict, importance_df) -> str:
    """Compose the prompt sent to the OpenAI API."""
    top_features = ""
    if importance_df is not None and not importance_df.empty:
        top10 = importance_df.head(10)
        top_features = "\n".join(
            f"  {i+1}. {row['feature']} ({row['importance']:.4f})"
            for i, (_, row) in enumerate(top10.iterrows())
        )

    prompt = f"""
You are analysing airline passenger satisfaction survey data.

Dataset overview:
- Total records: {stats.get('total_records', 'N/A'):,}
- Satisfaction rate: {stats.get('satisfaction_rate', 'N/A')}%
- Dissatisfaction rate: {stats.get('dissatisfaction_rate', 'N/A')}%

Model performance:
- Random Forest   → Accuracy: {rf_metrics.get('accuracy')}, F1: {rf_metrics.get('f1_score')}, AUC: {rf_metrics.get('roc_auc')}
- Logistic Regression → Accuracy: {lr_metrics.get('accuracy')}, F1: {lr_metrics.get('f1_score')}, AUC: {lr_metrics.get('roc_auc')}
- Best model: {comparison.get('best_model')}

Top 10 features driving satisfaction (Random Forest):
{top_features}

Write a professional Markdown insight report with these sections:
1. Executive Summary
2. Satisfaction Overview
3. Key Drivers of Passenger Satisfaction
4. Model Performance Comparison
5. Operational Insights for Airlines
6. Limitations and Future Improvements

Keep the report under 900 words. Use headers, bullet points, and tables where appropriate.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Rule-based report (no API key required)
# ---------------------------------------------------------------------------

def _rule_based_report(stats: dict, rf_metrics: dict, lr_metrics: dict,
                        comparison: dict, importance_df) -> str:
    """
    Generate a structured Markdown report from metrics using string
    templates — no external API call required.
    """
    today         = date.today().strftime("%B %d, %Y")
    best_model    = comparison.get("best_model", "N/A")
    sat_rate      = stats.get("satisfaction_rate", "N/A")
    dis_rate      = stats.get("dissatisfaction_rate", "N/A")
    total         = stats.get("total_records", "N/A")

    rf_acc   = rf_metrics.get("accuracy", "N/A")
    rf_prec  = rf_metrics.get("precision", "N/A")
    rf_rec   = rf_metrics.get("recall", "N/A")
    rf_f1    = rf_metrics.get("f1_score", "N/A")
    rf_auc   = rf_metrics.get("roc_auc", "N/A")

    lr_acc   = lr_metrics.get("accuracy", "N/A")
    lr_prec  = lr_metrics.get("precision", "N/A")
    lr_rec   = lr_metrics.get("recall", "N/A")
    lr_f1    = lr_metrics.get("f1_score", "N/A")
    lr_auc   = lr_metrics.get("roc_auc", "N/A")

    # Determine satisfaction insight language
    if isinstance(sat_rate, float):
        if sat_rate >= 55:
            sat_insight = (
                f"A majority of passengers ({sat_rate}%) reported being satisfied. "
                "However, the remaining dissatisfied segment still represents a "
                "significant business risk and warrants targeted improvement."
            )
        else:
            sat_insight = (
                f"A large proportion of passengers ({dis_rate}%) reported dissatisfaction. "
                "This signals systemic service gaps that require immediate operational attention."
            )
    else:
        sat_insight = "Satisfaction distribution data unavailable."

    # Format top features
    if importance_df is not None and not importance_df.empty:
        top10 = importance_df.head(10)
        feature_rows = "\n".join(
            f"| {i+1} | {row['feature'].replace('num__', '').replace('cat__', '').replace('_', ' ').title()} "
            f"| {row['importance']:.4f} |"
            for i, (_, row) in enumerate(top10.iterrows())
        )
        features_table = (
            "| Rank | Feature | Importance Score |\n"
            "|------|---------|------------------|\n"
            + feature_rows
        )
        # Key driver narrative
        top1 = top10.iloc[0]["feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title()
        top2 = top10.iloc[1]["feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title() if len(top10) > 1 else ""
        driver_narrative = (
            f"The most influential predictor of passenger satisfaction is **{top1}**. "
            f"{'**' + top2 + '** ranks second.' if top2 else ''} "
            "These features reflect core aspects of the in-flight and ground experience "
            "that most strongly differentiate satisfied from dissatisfied passengers."
        )
    else:
        features_table   = "_Feature importance data not available._"
        driver_narrative = "Feature importance could not be extracted for this run."

    # Customer type breakdowns (if available)
    extra_breakdowns = ""
    for col_key, label in [
        ("customer_type_satisfaction_rate",  "Customer Type"),
        ("type_of_travel_satisfaction_rate", "Type of Travel"),
        ("class_satisfaction_rate",          "Travel Class"),
    ]:
        breakdown = stats.get(col_key)
        if breakdown:
            rows = "\n".join(
                f"| {k.title()} | {round(v*100, 1)}% |"
                for k, v in breakdown.items()
            )
            extra_breakdowns += (
                f"\n### Satisfaction by {label}\n"
                f"| Category | Satisfaction Rate |\n"
                f"|----------|------------------|\n"
                f"{rows}\n"
            )

    report = f"""# Airline Passenger Satisfaction — AI Insight Report

**Generated:** {today}
**Dataset records analysed:** {total:,} passengers
**Report generated by:** Rule-based insight engine (no API key detected)

---

## 1. Executive Summary

This report presents the findings of an end-to-end machine learning analysis of airline passenger satisfaction survey data. Two classification models were trained and evaluated: a **Random Forest** and a **Logistic Regression**. The **{best_model}** achieved the best overall performance.

Key findings:
- **{sat_rate}%** of passengers reported satisfaction; **{dis_rate}%** reported dissatisfaction.
- The best model achieved an F1 score of **{rf_f1 if best_model == 'Random Forest' else lr_f1}** and AUC of **{rf_auc if best_model == 'Random Forest' else lr_auc}**.
- In-flight service attributes (Wi-Fi, entertainment, seat comfort) are the strongest drivers of satisfaction.

---

## 2. Satisfaction Overview

{sat_insight}

| Metric | Value |
|--------|-------|
| Total Passengers Analysed | {total:,} |
| Overall Satisfaction Rate | {sat_rate}% |
| Overall Dissatisfaction Rate | {dis_rate}% |

{extra_breakdowns}

---

## 3. Key Drivers of Passenger Satisfaction

{driver_narrative}

### Top 10 Features by Predictive Importance (Random Forest)

{features_table}

**Interpretation:**
- **In-flight service quality** (Wi-Fi, entertainment, food, comfort) consistently ranks among the top predictors.
- **Flight type and passenger class** significantly moderate satisfaction: business-class and business-travel passengers tend to have higher expectations and show distinct satisfaction patterns.
- **Departure and arrival delays** negatively impact satisfaction, though their effect is secondary to service quality ratings.

---

## 4. Model Performance Comparison

Two models were trained on an 80/20 stratified train-test split:

| Metric | Random Forest | Logistic Regression |
|--------|:------------:|:-------------------:|
| Accuracy  | {rf_acc}  | {lr_acc}  |
| Precision | {rf_prec} | {lr_prec} |
| Recall    | {rf_rec}  | {lr_rec}  |
| F1 Score  | {rf_f1}   | {lr_f1}   |
| ROC-AUC   | {rf_auc}  | {lr_auc}  |

**Winner: {best_model}**

The Random Forest benefits from its ensemble nature — it captures non-linear interactions between features that a linear model cannot represent. Logistic Regression provides a strong interpretable baseline and is competitive where relationships are approximately linear.

---

## 5. Operational Insights for Airlines

Based on model results and feature importance, the following operational actions are recommended:

1. **Invest in in-flight Wi-Fi quality.** Digital connectivity is a top satisfaction driver, particularly for business travellers.
2. **Improve seat comfort and legroom.** Physical comfort strongly predicts satisfaction across all travel classes.
3. **Enhance in-flight entertainment.** Content variety and system reliability are leading differentiators.
4. **Reduce delays.** Even moderate departure/arrival delays measurably reduce satisfaction scores.
5. **Tailor service by travel type.** Personal-travel passengers show different satisfaction profiles than business travellers — targeted communication and onboard experience adjustments can close gaps.
6. **Focus on Economy Class.** Satisfaction rates are typically lower in Economy; incremental improvements here affect the largest passenger segment.

---

## 6. Limitations and Future Improvements

### Limitations
- **Survey bias:** Satisfaction scores are self-reported and may reflect mood, recency, or cultural factors unrelated to service quality.
- **Static snapshot:** This analysis reflects a historical dataset; service quality and passenger demographics evolve over time.
- **No temporal modelling:** The dataset lacks timestamps, preventing trend analysis.
- **Imbalanced classes:** If one satisfaction class is significantly more common, model metrics may be inflated.

### Future Improvements
- **Real-time pipeline:** Deploy the model as a REST API that processes survey submissions as they arrive.
- **Deep learning exploration:** Neural networks (e.g., TabNet) may capture complex feature interactions more effectively.
- **SHAP / LIME explanations:** Add individual prediction explanations for customer service teams.
- **Segment-specific models:** Train separate models for Economy / Business / First Class passengers.
- **Longitudinal analysis:** Collect timestamped data to detect seasonal trends and measure improvement over time.
- **NLP on open-text feedback:** Mine unstructured comments for additional drivers not captured by numeric ratings.

---

*This report was auto-generated by the Airline Satisfaction Insight Analyzer pipeline.*
"""
    return report


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def generate_insight_report(
    stats: dict,
    rf_metrics: dict,
    lr_metrics: dict,
    comparison: dict,
    importance_df=None,
) -> Path:
    """
    Generate and save the AI insight report.

    Tries OpenAI API first; falls back to rule-based generation.
    Saves the report to outputs/reports/ai_insight_report.md.

    Returns
    -------
    Path to the saved report file.
    """
    print("\n[genai_insights] Generating insight report …")

    report_text = None

    # Attempt OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            print("[genai_insights] OPENAI_API_KEY detected — calling OpenAI API …")
            prompt      = _build_openai_prompt(stats, rf_metrics, lr_metrics, comparison, importance_df)
            report_text = _call_openai_api(prompt)
            print("[genai_insights] OpenAI API response received.")
        except Exception as exc:
            print(f"[genai_insights] OpenAI API call failed ({exc}). Falling back to rule-based report.")
            report_text = None

    # Rule-based fallback
    if report_text is None:
        print("[genai_insights] Generating rule-based insight report …")
        report_text = _rule_based_report(stats, rf_metrics, lr_metrics, comparison, importance_df)

    # Save to disk
    report_path = config.REPORTS_DIR / config.REPORT_FILENAME
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[genai_insights] Report saved → {report_path}")
    return report_path
