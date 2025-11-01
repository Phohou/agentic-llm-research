import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    FONT_SIZES,
    get_repo_color_mapping,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    # Filter to only include data up to 2025-07-31
    df = df[df["author_date"] < "2025-08-01"]
    return df


def create_sparklines(df, date_field, period="M"):
    """
    Create small multiple sparklines for each repository showing commit patterns.

    Args:
        df: DataFrame with commit data
        date_field: Column name for the date field to use
        period: Time period for aggregation ('W' for week, 'M' for month, 'Q' for quarter)
    """
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    # Create time period column
    df["time_period"] = df[date_field].dt.to_period(period)

    # Get unique repositories and sort by total commits (descending)
    repo_totals = df.groupby("repo_full_name").size().sort_values(ascending=False)
    repos = repo_totals.index.tolist()

    # Get color mapping for repositories
    color_map = get_repo_color_mapping(repos)

    # Calculate number of rows and columns for subplots
    n_repos = len(repos)
    n_cols = 1  # Single column for clean vertical layout
    n_rows = n_repos

    # Create figure with appropriate size
    fig_height = n_repos * 0.4  # Reduced height per sparkline for compactness
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(FIG_SIZE_SINGLE_COL[0], fig_height), sharex=True
    )

    # Ensure axes is always a list for consistent indexing
    if n_repos == 1:
        axes = [axes]

    # Create sparkline for each repository
    for idx, repo in enumerate(repos):
        ax = axes[idx]

        # Filter data for this repository
        repo_data = df[df["repo_full_name"] == repo].copy()

        # Group by time period and count commits
        commit_counts = (
            repo_data.groupby("time_period").size().reset_index(name="count")
        )

        # Convert period to timestamp for plotting
        commit_counts["timestamp"] = commit_counts["time_period"].dt.to_timestamp()

        # Fill in missing periods with zeros for continuity
        all_periods = pd.period_range(
            start=df["time_period"].min(),
            end=df["time_period"].max(),
            freq=period,
        )
        all_timestamps = all_periods.to_timestamp()

        # Create a complete series with zeros for missing periods
        complete_series = pd.DataFrame({"timestamp": all_timestamps})
        complete_series = complete_series.merge(
            commit_counts, on="timestamp", how="left"
        )
        complete_series["count"] = complete_series["count"].fillna(0)

        # Plot sparkline
        color = color_map.get(repo, "gray")
        ax.plot(
            complete_series["timestamp"],
            complete_series["count"],
            color=color,
            linewidth=1.0,
            alpha=0.8,
        )
        ax.fill_between(
            complete_series["timestamp"],
            complete_series["count"],
            alpha=0.2,
            color=color,
        )

        # Add repository name as y-axis label
        display_name = REPO_NAME_TO_DISPLAY.get(repo, repo)
        ax.set_ylabel(display_name, fontsize=FONT_SIZES["tick"], rotation=0, ha="right")

        # Remove all spines and ticks for minimal sparkline aesthetic
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", which="both", length=0, labelsize=FONT_SIZES["tick"])
        ax.set_yticks([])

        # Only show x-axis on the bottom subplot
        if idx < n_repos - 1:
            ax.tick_params(axis="x", which="both", length=0)
            ax.set_xticks([])
        else:
            ax.tick_params(
                axis="x", which="both", length=3, labelsize=FONT_SIZES["tick"]
            )
            ax.xaxis.set_major_locator(mdates.YearLocator())
            # Format x-axis labels to show year
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, left=0.25, right=0.95)

    return fig


def create_sparklines_monthly(df, date_field):
    """
    Create monthly sparklines showing commit patterns.
    """
    return create_sparklines(df, date_field, period="M")


def create_sparklines_weekly(df, date_field):
    """
    Create weekly sparklines showing commit patterns.
    """
    return create_sparklines(df, date_field, period="W")


def create_sparklines_quarterly(df, date_field):
    """
    Create quarterly sparklines showing commit patterns.
    """
    return create_sparklines(df, date_field, period="Q")


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    # Create monthly sparklines
    fig1 = create_sparklines_monthly(df, "author_date")
    fig1.savefig(
        outputDir / "commits_sparklines_monthly.png", dpi=300, bbox_inches="tight"
    )
    fig1.savefig(
        figuresDir / "rq1_commits_sparklines_monthly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig1)

    # Create weekly sparklines (more detailed)
    fig2 = create_sparklines_weekly(df, "author_date")
    fig2.savefig(
        outputDir / "commits_sparklines_weekly.png", dpi=300, bbox_inches="tight"
    )
    fig2.savefig(
        figuresDir / "rq1_commits_sparklines_weekly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig2)

    # Create quarterly sparklines (overview)
    fig3 = create_sparklines_quarterly(df, "author_date")
    fig3.savefig(
        outputDir / "commits_sparklines_quarterly.png", dpi=300, bbox_inches="tight"
    )
    fig3.savefig(
        figuresDir / "rq1_commits_sparklines_quarterly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig3)

    print(f"\nSparkline PNG files saved in: {outputDir}")
    print(f"Sparkline PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
