# Airline Passenger Satisfaction Insight Analyzer

An end-to-end AI-powered analytics system that processes airline passenger satisfaction survey data, trains machine learning models to predict satisfaction, and generates a natural-language executive insight report.

---

## Project Overview

Airlines collect large volumes of passenger satisfaction survey data, but extracting actionable insights quickly is challenging. This project builds a complete analysis pipeline that:

1. Loads and cleans the raw survey dataset
2. Performs exploratory data analysis with rich visualisations
3. Trains and compares two classification models (Random Forest & Logistic Regression)
4. Generates a professional AI-written insight report summarising findings

---

## Problem Definition

**Business Problem:** How can an airline rapidly identify the key drivers of passenger dissatisfaction and predict whether a passenger will report being satisfied — enabling proactive service improvements?

**ML Framing:** Binary classification — predict whether a passenger is `satisfied` (1) or `neutral / dissatisfied` (0) based on their flight profile and service ratings.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Maven Analytics / Kaggle |
| Link | [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) |
| Rows | ~130,000 passengers |
| Features | 22 (demographics, flight details, service ratings) |
| Target | `satisfaction` — binary (satisfied / neutral or dissatisfied) |

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Passenger gender |
| `customer_type` | Categorical | Loyal / disloyal customer |
| `age` | Numeric | Passenger age |
| `type_of_travel` | Categorical | Business / Personal |
| `class` | Categorical | Business / Economy / Economy Plus |
| `flight_distance` | Numeric | Distance of the flight (miles) |
| `inflight_wifi_service` | Numeric | Wi-Fi satisfaction rating (0–5) |
| `seat_comfort` | Numeric | Seat comfort rating (0–5) |
| `inflight_entertainment` | Numeric | Entertainment rating (0–5) |
| `departure_delay_in_minutes` | Numeric | Departure delay |
| `arrival_delay_in_minutes` | Numeric | Arrival delay |
| ... | ... | Additional service ratings |

---

## Project Architecture

```
airline-satisfaction-insight-analyzer/
│
├── data/
│   └── airline_passenger_satisfaction.csv   ← Place your dataset here
│
├── notebooks/
│   └── exploratory_analysis.ipynb           ← Interactive EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py            ← All paths, constants, hyper-parameters
│   ├── data_processing.py   ← Load, clean, encode, split data
│   ├── visualization.py     ← Generate & save all EDA figures
│   ├── model_training.py    ← Build, train, save both ML models
│   ├── evaluation.py        ← Metrics, confusion matrix, ROC, comparison
│   ├── genai_insights.py    ← AI or rule-based natural language report
│   └── utils.py             ← Helpers, timers, directory checks
│
├── outputs/
│   ├── figures/             ← PNG visualisations (auto-generated)
│   ├── reports/             ← ai_insight_report.md (auto-generated)
│   └── models/              ← Saved .joblib models + metrics JSON
│
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                  ← Single entry-point for the full pipeline
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/airline-satisfaction-insight-analyzer.git
cd airline-satisfaction-insight-analyzer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Download from: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
# Rename the file to: airline_passenger_satisfaction.csv
# Place it in: data/
```

---

## How to Run the Pipeline

### Full pipeline (recommended)

```bash
python main.py
```

This executes all six steps and prints output locations when finished.

### Custom dataset path

```bash
python main.py --data path/to/your_dataset.csv
```

### Skip visualisations (faster debug)

```bash
python main.py --skip-viz
```

### With OpenAI-powered report

```bash
export OPENAI_API_KEY="sk-..."
python main.py
```

If `OPENAI_API_KEY` is not set, a high-quality rule-based report is generated automatically.

### Interactive notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## Example Outputs

After a successful run:

```
outputs/
├── figures/
│   ├── 01_satisfaction_distribution.png
│   ├── 02_satisfaction_by_customer_type.png
│   ├── 03_satisfaction_by_travel_type.png
│   ├── 04_satisfaction_by_class.png
│   ├── 05_flight_distance_distribution.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_service_ratings.png
│   ├── 08_age_distribution.png
│   ├── 09_roc_curves.png
│   ├── 10_feature_importance.png
│   ├── 11_cm_random_forest.png
│   └── 12_cm_logistic_regression.png
│
├── reports/
│   └── ai_insight_report.md        ← Executive summary report
│
└── models/
    ├── random_forest_model.joblib
    ├── logistic_regression_model.joblib
    ├── rf_feature_importance.csv
    └── model_metrics.json
```

### Typical model performance

| Metric | Random Forest | Logistic Regression |
|--------|:------------:|:-------------------:|
| Accuracy | ~0.96 | ~0.87 |
| F1 Score | ~0.96 | ~0.87 |
| ROC-AUC  | ~0.99 | ~0.93 |

---

## Capstone Requirements

### Data Processing

| Requirement | Implementation |
|-------------|---------------|
| Data loading | `src/data_processing.py` → `load_dataset()` |
| Data cleaning | `clean_data()` — handles missing values, duplicates, type encoding |
| Column normalisation | `standardize_column_names()` — converts all names to snake_case |
| Data manipulation | Feature type detection, binary target encoding, median/mode imputation |
| Data visualisation | `src/visualization.py` — 8+ chart types saved to `outputs/figures/` |

### Machine Learning

| Requirement | Implementation |
|-------------|---------------|
| Classification model | RandomForestClassifier + LogisticRegression |
| Train/test split | 80/20 stratified split (`src/data_processing.py` → `split_data()`) |
| Preprocessing pipeline | `ColumnTransformer` with imputation, scaling, one-hot encoding |
| Model comparison | Side-by-side metrics table in `src/evaluation.py` → `compare_models()` |
| Evaluation metrics | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Feature importance | Extracted from Random Forest, saved to CSV |

### Generative AI

| Requirement | Implementation |
|-------------|---------------|
| AI-generated report | `src/genai_insights.py` → `generate_insight_report()` |
| OpenAI API integration | `_call_openai_api()` — uses `gpt-4o-mini` if API key present |
| Automatic fallback | `_rule_based_report()` — structured template report without API |
| Report sections | Executive Summary, Satisfaction Overview, Key Drivers, Model Performance, Operational Insights, Limitations |
| Output location | `outputs/reports/ai_insight_report.md` |

---

## Capstone Written Responses

### Problem Definition

Airlines collect millions of passenger satisfaction survey responses annually but struggle to rapidly extract actionable insights. Manual analysis is slow and inconsistent. This project addresses this by building an automated ML pipeline that:
- Identifies which service attributes most strongly predict satisfaction
- Enables comparison of different analytical approaches
- Generates plain-language reports accessible to non-technical stakeholders

### Approach and Tool Selection

**pandas / numpy** were chosen for data manipulation due to their industry-standard status and rich API for tabular data. **scikit-learn** provides a clean Pipeline API that bundles preprocessing and modelling, preventing data leakage and simplifying deployment. **Random Forest** was selected as the primary model because it naturally handles non-linear relationships and mixed feature types without extensive manual tuning. **Logistic Regression** serves as the interpretable baseline — its coefficients provide a sanity check on the Random Forest's findings. **OpenAI API** enables generative AI report writing; the rule-based fallback ensures the project works without an API key.

### Reflection

The project demonstrates that in-flight service quality (Wi-Fi, entertainment, seat comfort) is a far stronger predictor of satisfaction than operational factors (delays, distance). Business-class and business-travel passengers exhibit distinct satisfaction patterns from Economy leisure passengers, suggesting that segment-specific improvements would be more effective than one-size-fits-all interventions.

Key challenges included ensuring the pipeline is robust to minor dataset schema variations (handled via snake_case normalisation and flexible column resolution) and making the GenAI component gracefully degrade when no API key is available.

Future improvements would include real-time prediction via a REST API, SHAP explanations for individual passengers, and longitudinal trend analysis as timestamped data becomes available.

---

## Limitations

- Survey data is self-reported and may contain response bias
- The dataset is a historical snapshot — no temporal modelling is possible
- The rule-based insight report, while detailed, lacks the nuance of a large language model
- Class imbalance (if present) may inflate accuracy metrics

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| scikit-learn | 1.3+ | ML models & pipelines |
| matplotlib | 3.7+ | Base visualisation |
| seaborn | 0.12+ | Statistical visualisation |
| joblib | 1.3+ | Model persistence |
| openai | 1.0+ | GenAI report (optional) |

---

## License

MIT License — see `LICENSE` for details.

---

*Airline Satisfaction Insight Analyzer — Capstone Project*
