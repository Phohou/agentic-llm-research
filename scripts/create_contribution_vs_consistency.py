import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    FONT_SIZES,
    apply_grid_style,
    get_repo_color_mapping,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    return df


def calculate_coefficient_of_variation(df, date_field, period="M"):
    """
    Calculate coefficient of variation (CV) for commit frequency per repository.
    CV = (standard deviation / mean) * 100

    Args:
        df: DataFrame with commit data
        date_field: Column name for the date field to use
        period: Time period for aggregation ('W' for week, 'M' for month, 'Q' for quarter)

    Returns:
        DataFrame with CV statistics per repository
    """
    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    # Create time period column
    df["time_period"] = df[date_field].dt.to_period(period)

    # Get the full range of periods
    all_periods = pd.period_range(
        start=df["time_period"].min(),
        end=df["time_period"].max(),
        freq=period,
    )

    cv_stats = []

    for repo in df["repo_full_name"].unique():
        repo_data = df[df["repo_full_name"] == repo]

        # Count commits per period
        commit_counts = (
            repo_data.groupby("time_period").size().reset_index(name="count")
        )

        # Create complete series including periods with zero commits
        complete_data = pd.DataFrame({"time_period": all_periods})
        complete_data = complete_data.merge(commit_counts, on="time_period", how="left")
        complete_data["count"] = complete_data["count"].fillna(0)

        # Calculate statistics
        counts = complete_data["count"].values
        mean_commits = np.mean(counts)
        std_commits = np.std(counts, ddof=1)  # Sample standard deviation

        # Calculate CV
        if mean_commits > 0:
            cv = (std_commits / mean_commits) * 100
        else:
            cv = 0

        cv_stats.append(
            {
                "repo_full_name": repo,
                "mean_commits": mean_commits,
                "std_commits": std_commits,
                "cv": cv,
                "total_commits": len(repo_data),
            }
        )

    return pd.DataFrame(cv_stats)


def calculate_contribution_percentage(df):
    """
    Calculate the contribution percentage for each repository.

    Returns:
        DataFrame with contribution percentages
    """
    total_commits = len(df)
    contribution_stats = []

    for repo in df["repo_full_name"].unique():
        repo_commits = len(df[df["repo_full_name"] == repo])
        contribution_pct = (repo_commits / total_commits) * 100

        contribution_stats.append(
            {
                "repo_full_name": repo,
                "contribution_percentage": contribution_pct,
                "repo_commits": repo_commits,
            }
        )

    return pd.DataFrame(contribution_stats)


def create_contribution_vs_consistency_scatter(df, date_field, period="M"):
    """
    Create a scatter plot showing the relationship between contribution percentage
    and coefficient of variation (consistency).
    """
    setup_plotting_style()

    # Calculate CV statistics
    cv_stats = calculate_coefficient_of_variation(df, date_field, period)
    contribution_stats = calculate_contribution_percentage(df)
    analysis_df = cv_stats.merge(contribution_stats, on="repo_full_name")

    # Apply display names
    analysis_df["display_name"] = analysis_df["repo_full_name"].map(
        lambda x: REPO_NAME_TO_DISPLAY.get(x, x)
    )

    # Get color mapping
    color_map = get_repo_color_mapping(analysis_df["repo_full_name"].tolist())
    colors = [color_map.get(repo, "gray") for repo in analysis_df["repo_full_name"]]

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    # Create scatter plot with individual points for legend
    for idx, row in analysis_df.iterrows():
        ax.scatter(
            row["contribution_percentage"],
            row["cv"],
            c=color_map.get(row["repo_full_name"], "gray"),
            s=80,  # Reduced size for cleaner look
            edgecolors="white",
            linewidths=1.5,
            label=row["display_name"],
        )

    # Add median reference lines (optional but helpful)
    median_contrib = analysis_df["contribution_percentage"].median()
    median_cv = analysis_df["cv"].median()
    ax.axvline(
        median_contrib, color="gray", linestyle="--", alpha=0.3, linewidth=1, zorder=0
    )
    ax.axhline(
        median_cv, color="gray", linestyle="--", alpha=0.3, linewidth=1, zorder=0
    )

    # Customize axes
    ax.set_xlabel(
        "Contribution to Total Commits (%)", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylabel(
        "Coefficient of Variation (%)\n(Higher is less consistent)",
        fontsize=FONT_SIZES["axis_label"],
    )

    # Add legend
    from plot_utils import setup_legend

    setup_legend(ax, loc="upper right", max_width=15, ncol=2)

    # Apply grid style
    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    plt.tight_layout()

    return fig, analysis_df


def main():
    df = load_data()

    # Convert timezone if needed before filtering
    if df["author_date"].dt.tz is not None:
        df["author_date"] = df["author_date"].dt.tz_convert(None)

    # Filter data to only include commits up to 2025-07-31
    cutoff_date = pd.to_datetime("2025-07-31")
    df = df[df["author_date"] <= cutoff_date]

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    # Create monthly scatter plot
    fig, analysis_df = create_contribution_vs_consistency_scatter(
        df, "author_date", period="M"
    )

    fig.savefig(
        outputDir / "contribution_vs_consistency.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        figuresDir / "rq1_contribution_vs_consistency.pdf",
        bbox_inches="tight",
        format="pdf",
    )

    # Save the analysis data
    analysis_df.to_csv(outputDir / "contribution_vs_consistency_stats.csv", index=False)

    plt.close(fig)

    print(f"\nScatter plot PNG saved in: {outputDir}")
    print(f"Scatter plot PDF saved in: {figuresDir}")
    print(f"Analysis statistics CSV saved in: {outputDir}")

    # Print summary statistics
    print("\n=== Contribution vs. Consistency Analysis ===")
    print(
        analysis_df[["display_name", "contribution_percentage", "cv", "total_commits"]]
        .sort_values("contribution_percentage", ascending=False)
        .to_string(index=False)
    )

    # Calculate and print correlation
    correlation = analysis_df["contribution_percentage"].corr(analysis_df["cv"])
    print(f"\nCorrelation between contribution % and CV: {correlation:.3f}")


if __name__ == "__main__":
    main()
