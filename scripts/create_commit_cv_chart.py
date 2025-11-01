import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

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

    A lower CV indicates more consistent commit patterns.
    A higher CV indicates more variable/bursty commit patterns.

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
                "min_commits": np.min(counts),
                "max_commits": np.max(counts),
                "median_commits": np.median(counts),
            }
        )

    return pd.DataFrame(cv_stats)


def create_cv_chart(df, date_field, period="M"):
    """
    Create a bar chart showing coefficient of variation for each repository.

    Args:
        df: DataFrame with commit data
        date_field: Column name for the date field to use
        period: Time period for aggregation ('W' for week, 'M' for month, 'Q' for quarter)
    """
    setup_plotting_style()

    # Calculate CV statistics
    cv_stats = calculate_coefficient_of_variation(df, date_field, period)

    # Sort by CV (descending) to show most variable repositories at top
    cv_stats = cv_stats.sort_values("cv", ascending=True)

    # Apply display names
    cv_stats["display_name"] = cv_stats["repo_full_name"].map(
        lambda x: REPO_NAME_TO_DISPLAY.get(x, x)
    )

    # Get color mapping
    color_map = get_repo_color_mapping(cv_stats["repo_full_name"].tolist())
    colors = [color_map.get(repo, "gray") for repo in cv_stats["repo_full_name"]]

    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_SINGLE_COL[1]))

    # Create horizontal bar chart
    y_pos = np.arange(len(cv_stats))
    bars = ax.barh(y_pos, cv_stats["cv"], color=colors, alpha=0.8, edgecolor="none")

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cv_stats["display_name"], fontsize=FONT_SIZES["tick"])
    ax.set_xlabel(
        "Commits Coefficient of Variation (%)", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylabel("")

    # Apply grid style (only horizontal lines for horizontal bar chart)
    ax.grid(True, axis="x", linestyle="-", alpha=0.3, color="#CCCCCC", linewidth=0.5)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    for i, (idx, row) in enumerate(cv_stats.iterrows()):
        ax.text(
            row["cv"] + 2,
            i,
            f'{row["cv"]:.1f}',
            va="center",
            fontsize=FONT_SIZES["annotation"],
            color="black",
        )

    plt.tight_layout()

    return fig, cv_stats


def create_cv_chart_monthly(df, date_field):
    """
    Create monthly CV chart showing commit consistency.
    """
    return create_cv_chart(df, date_field, period="M")


def create_cv_chart_weekly(df, date_field):
    """
    Create weekly CV chart showing commit consistency.
    """
    return create_cv_chart(df, date_field, period="W")


def create_cv_chart_quarterly(df, date_field):
    """
    Create quarterly CV chart showing commit consistency.
    """
    return create_cv_chart(df, date_field, period="Q")


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    # Create monthly CV chart
    fig1, stats1 = create_cv_chart_monthly(df, "author_date")
    fig1.savefig(outputDir / "commits_cv_monthly.png", dpi=300, bbox_inches="tight")
    fig1.savefig(
        figuresDir / "rq1_commits_cv_monthly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    stats1.to_csv(outputDir / "commits_cv_stats_monthly.csv", index=False)
    plt.close(fig1)

    # Create weekly CV chart
    fig2, stats2 = create_cv_chart_weekly(df, "author_date")
    fig2.savefig(outputDir / "commits_cv_weekly.png", dpi=300, bbox_inches="tight")
    fig2.savefig(
        figuresDir / "rq1_commits_cv_weekly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    stats2.to_csv(outputDir / "commits_cv_stats_weekly.csv", index=False)
    plt.close(fig2)

    # Create quarterly CV chart
    fig3, stats3 = create_cv_chart_quarterly(df, "author_date")
    fig3.savefig(outputDir / "commits_cv_quarterly.png", dpi=300, bbox_inches="tight")
    fig3.savefig(
        figuresDir / "rq1_commits_cv_quarterly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    stats3.to_csv(outputDir / "commits_cv_stats_quarterly.csv", index=False)
    plt.close(fig3)

    print(f"\nCV chart PNG files saved in: {outputDir}")
    print(f"CV chart PDF files saved in: {figuresDir}")
    print(f"CV statistics CSV files saved in: {outputDir}")

    # Print summary statistics
    print("\n=== Monthly CV Statistics ===")
    print(
        stats1[["display_name", "cv", "mean_commits", "std_commits"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
