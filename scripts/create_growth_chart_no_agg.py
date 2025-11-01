import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.interpolate import PchipInterpolator

from plot_utils import (
    PLOT_LINE_WIDTH_THIN,
    setup_plotting_style,
    MAIN_COLORS,
    FIG_SIZE_SINGLE_COL,
    PLOT_LINE_WIDTH,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def loadDf():
    """Loads the dataset from the parquet file."""

    dataPath = DATA_DIR / "issues.parquet"

    df = pd.read_parquet(dataPath)

    return df


def createDailyGrowthPlot(df, date):
    """Creates and saves a cumulative growth chart without aggregation."""
    setup_plotting_style()

    df = df.copy()
    if df[date].dt.tz is not None:
        df[date] = df[date].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    repos = df["repo"].unique()
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo"] == repo].copy()
        repo_df = repo_df.sort_values(date)

        # Calculate cumulative count for each issue
        repo_df["cumulative"] = range(1, len(repo_df) + 1)

        # Add starting point at zero
        start_date = repo_df[date].iloc[0]
        dates = pd.concat([pd.Series([start_date]), repo_df[date]], ignore_index=True)
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

    ax.set_ylabel("Number of Issues", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))

    # Show only year as major ticks (2019, 2020, etc.)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Add monthly minor ticks for grid lines (11 lines between Januaries)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # No rotation needed for years only
    plt.xticks(rotation=0)

    # Place legend on the right outside plot area
    setup_legend(
        ax,
        loc="upper left",
        # bbox_to_anchor=(1.01, 1),
        max_width=10,
    )

    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def plot_average_time_to_close_over_time(df, date_created, date_closed):
    """Plot the average time to close issues over time."""
    setup_plotting_style()

    plot_df = df.copy()
    plot_df[date_created] = pd.to_datetime(plot_df[date_created])
    plot_df[date_closed] = pd.to_datetime(plot_df[date_closed])

    plot_df["time_to_close_days"] = (
        plot_df[date_closed] - plot_df[date_created]
    ).dt.total_seconds() / (24 * 60 * 60)

    plot_df = plot_df[
        (plot_df["time_to_close_days"] >= 0) & (plot_df["time_to_close_days"] <= 730)
    ]

    plot_df["year_quarter"] = plot_df[date_created].dt.to_period("Q")

    quarterly_avg = (
        plot_df.groupby(["repo", "year_quarter"])["time_to_close_days"]
        .agg(["mean", "count"])
        .reset_index()
    )
    quarterly_avg["date"] = quarterly_avg["year_quarter"].dt.to_timestamp()
    quarterly_avg = quarterly_avg[quarterly_avg["count"] >= 5]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    repos = quarterly_avg["repo"].unique()

    # Get consistent color mapping
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_data = quarterly_avg[quarterly_avg["repo"] == repo]
        ax.plot(
            repo_data["date"],
            repo_data["mean"],
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH,
        )

    ax.set_xlabel("Date", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Average Time to Close (Days)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))

    # Show January of each year as major ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Add monthly minor ticks for grid lines
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)

    setup_legend(ax, loc="upper left")
    apply_grid_style(ax)
    plt.tight_layout()

    return fig


def plot_overall_average_time_to_close(df, date_created, date_closed):
    """Plot overall average time to close across all repositories."""
    setup_plotting_style()

    plot_df = df.copy()
    plot_df[date_created] = pd.to_datetime(plot_df[date_created])
    plot_df[date_closed] = pd.to_datetime(plot_df[date_closed])

    plot_df["time_to_close_days"] = (
        plot_df[date_closed] - plot_df[date_created]
    ).dt.total_seconds() / (24 * 60 * 60)

    plot_df = plot_df[
        (plot_df["time_to_close_days"] >= 0) & (plot_df["time_to_close_days"] <= 730)
    ]

    plot_df["year_quarter"] = plot_df[date_created].dt.to_period("Q")

    quarterly_avg = (
        plot_df.groupby("year_quarter")["time_to_close_days"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    quarterly_avg["date"] = quarterly_avg["year_quarter"].dt.to_timestamp()
    quarterly_avg = quarterly_avg[quarterly_avg["count"] >= 10]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    ax.plot(
        quarterly_avg["date"],
        quarterly_avg["mean"],
        label="Average (Mean)",
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH * 1.5,
    )
    ax.plot(
        quarterly_avg["date"],
        quarterly_avg["median"],
        label="Median",
        color=MAIN_COLORS[1],
        linewidth=PLOT_LINE_WIDTH,
    )

    ax.set_xlabel("Date", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Time to Close (Days)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))

    # Show January of each year as major ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Add monthly minor ticks for grid lines
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)

    setup_legend(ax, loc="upper left")
    apply_grid_style(ax)
    plt.tight_layout()

    return fig


def percentage_labeled_issues(df, repo_col):
    """Create a bar chart showing percentage of labeled issues by repository."""
    setup_plotting_style()

    repo_stats = []
    for repo in df[repo_col].unique():
        repo_data = df[df[repo_col] == repo]
        total_issues = len(repo_data)
        labeled_issues = len(repo_data[repo_data["labels_count"] >= 1])
        percentage = (labeled_issues / total_issues) * 100 if total_issues > 0 else 0

        repo_stats.append(
            {
                "repo": repo,
                "total_issues": total_issues,
                "labeled_issues": labeled_issues,
                "percentage": percentage,
            }
        )

    stats_df = pd.DataFrame(repo_stats).sort_values("percentage", ascending=True)
    cleaned_labels = [REPO_NAME_TO_DISPLAY.get(repo, repo) for repo in stats_df["repo"]]

    # Get consistent color mapping
    repo_colors = get_repo_color_mapping(stats_df["repo"])

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    bars = ax.barh(
        cleaned_labels,
        stats_df["percentage"].values,
        color=[repo_colors[repo] for repo in stats_df["repo"]],
    )

    for bar, percentage, labeled, total in zip(
        bars,
        stats_df["percentage"].values,
        stats_df["labeled_issues"].values,
        stats_df["total_issues"].values,
    ):
        label = f"{percentage:.1f}% ({labeled}/{total})"
        ax.text(
            percentage + 2,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=FONT_SIZES["annotation"],
            fontweight="bold",
        )

    ax.set_xlabel(
        "Percentage of Issues with Labels (%)", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylabel("Repository", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(0, max(stats_df["percentage"].values) + 15)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    apply_grid_style(ax)
    plt.tight_layout()

    return fig


def createDailyIssuesCount(df, date):
    """Creates a chart showing new issues per month with PCHIP smoothing."""
    setup_plotting_style()

    df = df.copy()
    if df[date].dt.tz is not None:
        df[date] = df[date].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    repos = df["repo"].unique()
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo"] == repo].copy()

        # Group by month and count issues per month
        repo_df["year_month"] = repo_df[date].dt.to_period("M")
        monthly_counts = (
            repo_df.groupby("year_month")
            .size()
            .reset_index(name="count")
        )
        monthly_counts["date"] = monthly_counts["year_month"].dt.to_timestamp()
        monthly_counts = monthly_counts.sort_values("date")

        # Apply PCHIP interpolation for smooth curves
        x = np.array([(d - monthly_counts["date"].iloc[0]).total_seconds() 
                      for d in monthly_counts["date"]])
        y = monthly_counts["count"].values
        
        # Create PCHIP interpolator
        pchip = PchipInterpolator(x, y)
        
        # Generate smooth curve
        x_smooth = np.linspace(x.min(), x.max(), min(300, len(x) * 3))
        y_smooth = pchip(x_smooth)
        
        # Ensure no negative values
        y_smooth = np.maximum(y_smooth, 0)
        
        # Convert back to dates
        dates_smooth = [monthly_counts["date"].iloc[0] + pd.Timedelta(seconds=float(xs)) 
                       for xs in x_smooth]

        ax.plot(
            dates_smooth,
            y_smooth,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel(
        "New Issues", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=0)

    setup_legend(ax, loc="upper left", max_width=10)
    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def createCumulativeGrowthPlot(df, date):
    """Creates and saves a cumulative growth chart."""
    setup_plotting_style()

    plot_df = df.copy()
    plot_df[date] = pd.to_datetime(plot_df[date])
    if plot_df[date].dt.tz is not None:
        plot_df[date] = plot_df[date].dt.tz_convert(None)

    plot_df = plot_df.sort_values(date)
    cutoff = pd.to_datetime("2025-07-31")
    plot_df = plot_df[plot_df[date] <= cutoff]
    plot_df["year_quarter"] = plot_df[date].dt.to_period("Q")

    quarterly_counts = (
        plot_df.groupby("year_quarter").size().reset_index(name="issue_count")
    )
    quarterly_counts["cumulative_count"] = quarterly_counts["issue_count"].cumsum()
    quarterly_counts["date"] = quarterly_counts["year_quarter"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    first_date = quarterly_counts["date"].iloc[0]
    start_point = pd.DataFrame({"date": [first_date], "cumulative_count": [0]})
    plot_data = pd.concat(
        [start_point, quarterly_counts[["date", "cumulative_count"]]], ignore_index=True
    )

    ax.plot(
        plot_data["date"],
        plot_data["cumulative_count"],
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH * 2,
    )

    ax.set_xlabel("Quarter", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(
        "Cumulative Number of Closed Issues", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, pos: f"{mdates.num2date(x).year} Q{(mdates.num2date(x).month - 1) // 3 + 1}"
        )
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.xticks(rotation=45)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
    )

    apply_grid_style(ax)
    plt.tight_layout()

    return fig


def main():
    try:
        df = loadDf()
        if df is None:
            return

        outputDir = Path("output")
        figuresDir = Path("figures")
        outputDir.mkdir(parents=True, exist_ok=True)
        figuresDir.mkdir(parents=True, exist_ok=True)

        print("\nCreating daily growth chart...")
        fig1 = createDailyGrowthPlot(df, "issue_created_at")
        fig1.savefig(outputDir / "daily_growth_chart.png", dpi=300, bbox_inches="tight")
        fig1.savefig(
            figuresDir / "rq1_daily_issue_trend_.pdf", bbox_inches="tight", format="pdf"
        )
        plt.close(fig1)

        print("\nCreating average time to close by repo chart...")
        fig3 = plot_average_time_to_close_over_time(
            df, "issue_created_at", "issue_closed_at"
        )
        fig3.savefig(
            outputDir / "average_time_to_close_by_repo2.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            figuresDir / "average_time_to_close_by_repo2.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close(fig3)

        print("\nCreating overall average time to close chart...")
        fig4 = plot_overall_average_time_to_close(
            df, "issue_created_at", "issue_closed_at"
        )
        fig4.savefig(
            outputDir / "overall_average_time_to_close2.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            figuresDir / "overall_average_time_to_close2.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close(fig4)

        print("\nCreating percentage labeled issues chart...")
        fig5 = percentage_labeled_issues(df, "repo")
        fig5.savefig(
            outputDir / "percentage_labeled_issues2.png", dpi=300, bbox_inches="tight"
        )
        fig5.savefig(
            figuresDir / "percentage_labeled_issues2.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close(fig5)

        print("\nCreating cumulative quarterly growth chart...")
        fig6 = createCumulativeGrowthPlot(df, "issue_created_at")
        fig6.savefig(
            outputDir / "cumulative_quarterly_growth2.png", dpi=300, bbox_inches="tight"
        )
        fig6.savefig(
            figuresDir / "cumulative_quarterly_growth2.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close(fig6)

        print("\nCreating daily issues count chart...")
        fig7 = createDailyIssuesCount(df, "issue_created_at")
        fig7.savefig(outputDir / "daily_issues_count.png", dpi=300, bbox_inches="tight")
        fig7.savefig(
            figuresDir / "rq1_daily_issues.pdf", bbox_inches="tight", format="pdf"
        )
        plt.close(fig7)

        print(f"PNG files saved in: {outputDir}")
        print(f"PDF files saved in: {figuresDir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# TODO:
# Get a quarterly dataset or create a function that plots it by quarter in order to smooth out the curves
# Fix the 0 start issue from the y axis plot
# Properly label the axes and add a title
# Can use 1k.... to save space on labeling if needed if there is a lot of data to plot at hand
# Style the plot properly
# Change legend names to be proper repo names instead of full github repo names (can add the the legend to blank spots of the graph)
# Consult the files that use the ieee import
# Use pointers on points of interest on the graph if needed, better to do it on the graph (annoying to do in code so can do it in post if needed)

# Pull request
# Look at how long issues are taken to be fixed
# Look at the average time an issue takes to be fixed
# Want to see which repos are contributing more in the first half or other half *
# Tie issue count with how long each repository is taking to close their issues (finding correlation)
# Look at the contributors (see if the more popular repos have more contributors or not)
# Find the percentage of issues that have labels can split the data for a study (if the labels are similar gentrify it)
# Can look at which type of isses have more replies (bug, feature request, etc)
# Look at what the feature requests are (use use llm to summarize the feature requests, fine tune and structure the output, watch for the words it tries to use)
# Clustering the data (can use the embeddings from openai to cluster the data)

# Exact unique values for labels get a set for all the labels gentrify them and merge the similar ones
# Get the data to be more readable
# Look at the size of the pr fix, get the diff file that has line deleted, inserted, replaced, etc.
# See if there is a correlation over the complexity of the bugs and the time it takes over time

# Remove the title
# Thicker lines
# Change figure size
