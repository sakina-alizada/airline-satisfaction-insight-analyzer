import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for script runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src import config

warnings.filterwarnings("ignore")

# Seaborn global theme
sns.set_theme(style="whitegrid", palette=config.PALETTE_MAIN, font_scale=1.1)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, filename: str):
    """Save *fig* to outputs/figures/<filename>.png and close it."""
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Satisfaction distribution (bar chart)
# ---------------------------------------------------------------------------

def plot_satisfaction_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing count of satisfied vs dissatisfied passengers."""
    target = config.TARGET_COLUMN
    labels = {1: "Satisfied", 0: "Neutral / Dissatisfied"}
    counts = df[target].value_counts().rename(index=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=config.PALETTE_BINARY[::-1], edgecolor="white", linewidth=0.8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{bar.get_height():,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Passenger Satisfaction Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Satisfaction Class", fontsize=12)
    ax.set_ylabel("Number of Passengers", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "01_satisfaction_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Satisfaction by customer type
# ---------------------------------------------------------------------------

def plot_satisfaction_by_customer_type(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: satisfaction rate per customer type."""
    col = "customer_type"
    if col not in df.columns:
        print(f"[visualization] Column '{col}' not found – skipping.")
        return None

    target = config.TARGET_COLUMN
    grouped = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )
    grouped[target] = grouped[target].map({1: "Satisfied", 0: "Neutral/Dissatisfied"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=grouped, x=col, y="Percentage",
        hue=target, palette=config.PALETTE_BINARY[::-1], ax=ax,
    )
    ax.set_title("Satisfaction Rate by Customer Type", fontsize=15, fontweight="bold")
    ax.set_xlabel("Customer Type", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(title="Satisfaction", fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
    fig.tight_layout()
    _save(fig, "02_satisfaction_by_customer_type.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Satisfaction by type of travel
# ---------------------------------------------------------------------------

def plot_satisfaction_by_travel_type(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: satisfaction rate per travel type."""
    col = "type_of_travel"
    if col not in df.columns:
        print(f"[visualization] Column '{col}' not found – skipping.")
        return None

    target = config.TARGET_COLUMN
    grouped = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )
    grouped[target] = grouped[target].map({1: "Satisfied", 0: "Neutral/Dissatisfied"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=grouped, x=col, y="Percentage",
        hue=target, palette=config.PALETTE_BINARY[::-1], ax=ax,
    )
    ax.set_title("Satisfaction Rate by Type of Travel", fontsize=15, fontweight="bold")
    ax.set_xlabel("Type of Travel", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(title="Satisfaction", fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
    fig.tight_layout()
    _save(fig, "03_satisfaction_by_travel_type.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Satisfaction by class
# ---------------------------------------------------------------------------

def plot_satisfaction_by_class(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: satisfaction rate per travel class."""
    col = "class"
    if col not in df.columns:
        print(f"[visualization] Column '{col}' not found – skipping.")
        return None

    target = config.TARGET_COLUMN
    grouped = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )
    grouped[target] = grouped[target].map({1: "Satisfied", 0: "Neutral/Dissatisfied"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=grouped, x=col, y="Percentage",
        hue=target, palette=config.PALETTE_BINARY[::-1], ax=ax,
    )
    ax.set_title("Satisfaction Rate by Travel Class", fontsize=15, fontweight="bold")
    ax.set_xlabel("Travel Class", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(title="Satisfaction", fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
    fig.tight_layout()
    _save(fig, "04_satisfaction_by_class.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Flight distance distribution
# ---------------------------------------------------------------------------

def plot_flight_distance_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE of flight distance, split by satisfaction."""
    col = "flight_distance"
    if col not in df.columns:
        print(f"[visualization] Column '{col}' not found – skipping.")
        return None

    target = config.TARGET_COLUMN
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, grp_df in df.groupby(target):
        label_str = "Satisfied" if label == 1 else "Neutral/Dissatisfied"
        color = config.PALETTE_BINARY[label]
        sns.histplot(
            grp_df[col], kde=True, bins=40,
            label=label_str, color=color, alpha=0.55, ax=ax,
        )
    ax.set_title("Flight Distance Distribution by Satisfaction", fontsize=15, fontweight="bold")
    ax.set_xlabel("Flight Distance (miles)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Satisfaction", fontsize=10)
    fig.tight_layout()
    _save(fig, "05_flight_distance_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of Pearson correlations for all numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# 7. Service rating distributions
# ---------------------------------------------------------------------------

def plot_service_ratings(df: pd.DataFrame) -> plt.Figure:
    """
    Box-plots of all service-rating columns grouped by satisfaction.
    Service columns are detected by checking for integer columns in 0–5 range
    that contain words like 'service', 'comfort', 'cleanliness', etc.
    """
    target = config.TARGET_COLUMN
    rating_keywords = [
        "inflight", "food", "service", "comfort", "cleanliness",
        "entertainment", "wifi", "boarding", "baggage", "checkin",
        "legroom", "seat", "gate", "departure", "arrival",
    ]
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    # Candidate: numeric column whose name contains a rating keyword
    rating_cols = [
        c for c in num_cols
        if any(kw in c for kw in rating_keywords) and c != target
    ]
    if not rating_cols:
        print("[visualization] No service rating columns detected – skipping.")
        return None

    plot_df = df[rating_cols + [target]].copy()
    plot_df[target] = plot_df[target].map({1: "Satisfied", 0: "Neutral/Dissatisfied"})
    melted  = plot_df.melt(id_vars=target, var_name="Service", value_name="Rating")

    # Clean up column names for display
    melted["Service"] = melted["Service"].str.replace("_", " ").str.title()

    n_cols = 3
    n_rows = (len(rating_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(rating_cols):
        ax = axes[i]
        col_label = col.replace("_", " ").title()
        sub = df[[col, target]].copy()
        sub[target] = sub[target].map({1: "Satisfied", 0: "Neutral/Dissatisfied"})
        sns.boxplot(
            data=sub, x=target, y=col,
            palette=config.PALETTE_BINARY[::-1], ax=ax,
        )
        ax.set_title(col_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Rating")
        ax.tick_params(axis="x", labelrotation=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Service Ratings by Satisfaction Class", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "07_service_ratings.png")
    return fig


# ---------------------------------------------------------------------------
# 8. Age distribution
# ---------------------------------------------------------------------------

def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of passenger ages split by satisfaction."""
    col = "age"
    if col not in df.columns:
        print(f"[visualization] Column '{col}' not found – skipping.")
        return None

    target = config.TARGET_COLUMN
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, grp_df in df.groupby(target):
        label_str = "Satisfied" if label == 1 else "Neutral/Dissatisfied"
        color = config.PALETTE_BINARY[label]
        sns.histplot(
            grp_df[col], kde=True, bins=30,
            label=label_str, color=color, alpha=0.55, ax=ax,
        )
    ax.set_title("Passenger Age Distribution by Satisfaction", fontsize=15, fontweight="bold")
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Satisfaction", fontsize=10)
    fig.tight_layout()
    _save(fig, "08_age_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# Master function — called from main.py
# ---------------------------------------------------------------------------

def generate_all_visualizations(df: pd.DataFrame):
    """Run all visualisation functions and print a summary."""
    print("\n[visualization] Generating all visualisations …")
    plot_satisfaction_distribution(df)
    plot_satisfaction_by_customer_type(df)
    plot_satisfaction_by_travel_type(df)
    plot_satisfaction_by_class(df)
    plot_flight_distance_distribution(df)
    plot_correlation_heatmap(df)
    plot_service_ratings(df)
    plot_age_distribution(df)
    print("[visualization] All visualisations saved to:", config.FIGURES_DIR)
