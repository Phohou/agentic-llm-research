import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

from plot_utils import (
    PLOT_LINE_WIDTH_THIN,
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    return df


def create_daily_commits_trend(df, date_field):
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    repos = df["repo_full_name"].unique()
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()
        repo_df = repo_df.sort_values(date_field)

        # Calculate cumulative count for each commit
        repo_df["cumulative"] = range(1, len(repo_df) + 1)

        # Add starting point at zero
        start_date = repo_df[date_field].iloc[0]
        dates = pd.concat(
            [pd.Series([start_date]), repo_df[date_field]], ignore_index=True
        )
        cumulative = pd.concat(
            [pd.Series([0]), pd.Series(repo_df["cumulative"].values)], ignore_index=True
        )

        ax.plot(
            dates,
            cumulative,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel("Number of Commits", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=0)

    setup_legend(
        ax,
        loc="upper left",
        max_width=15,
    )

    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def create_daily_commits_count(df, date_field):
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    repos = df["repo_full_name"].unique()
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()

        # Group by month and count commits per month
        repo_df["year_month"] = repo_df[date_field].dt.to_period("M")
        monthly_counts = repo_df.groupby("year_month").size().reset_index(name="count")
        monthly_counts["date"] = monthly_counts["year_month"].dt.to_timestamp()
        monthly_counts = monthly_counts.sort_values("date")

        # Apply PCHIP interpolation for smooth curves
        x = np.array(
            [
                (d - monthly_counts["date"].iloc[0]).total_seconds()
                for d in monthly_counts["date"]
            ]
        )
        y = monthly_counts["count"].values

        # Create PCHIP interpolator
        pchip = PchipInterpolator(x, y)

        # Generate smooth curve
        x_smooth = np.linspace(x.min(), x.max(), min(300, len(x) * 3))
        y_smooth = pchip(x_smooth)

        # Ensure no negative values
        y_smooth = np.maximum(y_smooth, 0)

        # Convert back to dates
        dates_smooth = [
            monthly_counts["date"].iloc[0] + pd.Timedelta(seconds=float(xs))
            for xs in x_smooth
        ]

        ax.plot(
            dates_smooth,
            y_smooth,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel("New Commits", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=0)

    setup_legend(
        ax,
        loc="upper left",
        max_width=15,
    )

    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def main():

    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    fig1 = create_daily_commits_trend(df, "author_date")
    fig1.savefig(outputDir / "daily_commits_trend.png", dpi=300, bbox_inches="tight")
    fig1.savefig(
        figuresDir / "rq1_commit_trend_daily.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig1)

    fig2 = create_daily_commits_count(df, "author_date")
    fig2.savefig(outputDir / "daily_commits_count.png", dpi=300, bbox_inches="tight")
    fig2.savefig(
        figuresDir / "rq1_daily_commits.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig2)

    print(f"\nPNG files saved in: {outputDir}")
    print(f"PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
