import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.ticker import FuncFormatter
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
    MAIN_COLORS,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def format_kmb(x, pos):
    """Format numbers with K, M, B suffixes."""
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    elif x >= 1e6:
        return f"{x/1e6:.0f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    else:
        return f"{x:.0f}"


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    return df


def create_cumulative_stats_trend(df, date_field):
    """Create cumulative trends for file changes, additions, and deletions (all repos combined)."""
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    # Sort by date
    df = df.sort_values(date_field)

    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    # Group by month and sum
    df["year_month"] = df[date_field].dt.to_period("M")

    monthly_files = df.groupby("year_month")["files_changed"].sum().reset_index()
    monthly_files.columns = ["year_month", "value"]
    monthly_files["date"] = monthly_files["year_month"].dt.to_timestamp()
    monthly_files = monthly_files.sort_values("date")

    monthly_insertions = df.groupby("year_month")["insertions"].sum().reset_index()
    monthly_insertions.columns = ["year_month", "value"]
    monthly_insertions["date"] = monthly_insertions["year_month"].dt.to_timestamp()
    monthly_insertions = monthly_insertions.sort_values("date")

    monthly_deletions = df.groupby("year_month")["deletions"].sum().reset_index()
    monthly_deletions.columns = ["year_month", "value"]
    monthly_deletions["date"] = monthly_deletions["year_month"].dt.to_timestamp()
    monthly_deletions = monthly_deletions.sort_values("date")

    # Calculate cumulative sums
    files_cumsum = monthly_files["value"].cumsum()
    insertions_cumsum = monthly_insertions["value"].cumsum()
    deletions_cumsum = monthly_deletions["value"].cumsum()

    # Add starting points at zero
    start_date = monthly_files["date"].iloc[0]
    dates = pd.concat(
        [pd.Series([start_date]), monthly_files["date"]], ignore_index=True
    )

    files_values = pd.concat(
        [pd.Series([1]), pd.Series(files_cumsum.values)], ignore_index=True
    )
    insertions_values = pd.concat(
        [pd.Series([1]), pd.Series(insertions_cumsum.values)], ignore_index=True
    )
    deletions_values = pd.concat(
        [pd.Series([1]), pd.Series(deletions_cumsum.values)], ignore_index=True
    )

    # Create second y-axis
    ax2 = ax1.twinx()

    # Plot insertions and deletions on left axis (ax1)
    line1 = ax1.plot(
        dates,
        insertions_values,
        label="Insertions",
        color=MAIN_COLORS[2],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )
    line2 = ax1.plot(
        dates,
        deletions_values,
        label="Deletions",
        color=MAIN_COLORS[3],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )

    # Plot files changed on right axis (ax2)
    line3 = ax2.plot(
        dates,
        files_values,
        label="Files Changed",
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )

    # Set log scale
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # Set labels
    ax1.set_ylabel("Insertions/Deletions", fontsize=FONT_SIZES["axis_label"])
    ax2.set_ylabel("Files Changed", fontsize=FONT_SIZES["axis_label"])

    # Format y-axes with K/M/B
    ax1.yaxis.set_major_formatter(FuncFormatter(format_kmb))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_kmb))

    # Set x-axis
    ax1.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    # Combine legends - need to manually set the legend with all lines
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]

    # Create legend with all lines explicitly
    legend = ax1.legend(
        lines,
        labels,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.8,
        borderpad=0.4,
        handlelength=1,
        fontsize=FONT_SIZES["legend"],
    )
    plt.setp(legend.get_texts(), fontsize=FONT_SIZES["legend"], color="#333333")

    apply_grid_style(ax1, major_alpha=0.6, minor_alpha=0.4)

    # Disable grid on ax2 to avoid overlapping
    ax2.grid(False)

    return fig


def create_daily_stats_trend(df, date_field):
    """Create monthly trends for file changes, additions, and deletions (all repos combined)."""
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    # Group by month and sum for all repos
    df["year_month"] = df[date_field].dt.to_period("M")

    monthly_files = df.groupby("year_month")["files_changed"].sum().reset_index()
    monthly_files.columns = ["year_month", "value"]
    monthly_files["date"] = monthly_files["year_month"].dt.to_timestamp()
    monthly_files = monthly_files.sort_values("date")
    monthly_files["value"] = monthly_files["value"].replace(
        0, 1
    )  # Replace 0 with 1 for log scale

    monthly_insertions = df.groupby("year_month")["insertions"].sum().reset_index()
    monthly_insertions.columns = ["year_month", "value"]
    monthly_insertions["date"] = monthly_insertions["year_month"].dt.to_timestamp()
    monthly_insertions = monthly_insertions.sort_values("date")
    monthly_insertions["value"] = monthly_insertions["value"].replace(0, 1)

    monthly_deletions = df.groupby("year_month")["deletions"].sum().reset_index()
    monthly_deletions.columns = ["year_month", "value"]
    monthly_deletions["date"] = monthly_deletions["year_month"].dt.to_timestamp()
    monthly_deletions = monthly_deletions.sort_values("date")
    monthly_deletions["value"] = monthly_deletions["value"].replace(0, 1)

    # Create second y-axis
    ax2 = ax1.twinx()

    # Apply PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for monotonic smoothing
    def smooth_data_pchip(dates, values, n_points=300):
        """Apply PCHIP interpolation for smooth, monotonic curves that preserve shape."""
        x = np.array([(d - dates.iloc[0]).total_seconds() for d in dates])
        y = np.array(values)

        # Create PCHIP interpolator
        pchip = PchipInterpolator(x, y)

        # Generate smooth curve
        x_smooth = np.linspace(x.min(), x.max(), n_points)
        y_smooth = pchip(x_smooth)

        # Ensure no negative values for log scale
        y_smooth = np.maximum(y_smooth, y.min())

        # Convert back to dates
        dates_smooth = [
            dates.iloc[0] + pd.Timedelta(seconds=float(xs)) for xs in x_smooth
        ]

        return dates_smooth, y_smooth

    # Smooth the data with PCHIP
    dates_ins_smooth, insertions_smooth = smooth_data_pchip(
        monthly_insertions["date"], monthly_insertions["value"]
    )
    dates_del_smooth, deletions_smooth = smooth_data_pchip(
        monthly_deletions["date"], monthly_deletions["value"]
    )
    dates_files_smooth, files_smooth = smooth_data_pchip(
        monthly_files["date"], monthly_files["value"]
    )

    # Plot insertions and deletions on left axis (ax1)
    line1 = ax1.plot(
        dates_ins_smooth,
        insertions_smooth,
        label="Insertions",
        color=MAIN_COLORS[2],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )
    line2 = ax1.plot(
        dates_del_smooth,
        deletions_smooth,
        label="Deletions",
        color=MAIN_COLORS[3],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )

    # Plot files changed on right axis (ax2)
    line3 = ax2.plot(
        dates_files_smooth,
        files_smooth,
        label="Files Changed",
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH_THIN,
    )

    # Set log scale
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # Set labels
    ax1.set_ylabel("New Insertions/Deletions", fontsize=FONT_SIZES["axis_label"])
    ax2.set_ylabel("New Files Changed", fontsize=FONT_SIZES["axis_label"])

    # Format y-axes with K/M/B
    ax1.yaxis.set_major_formatter(FuncFormatter(format_kmb))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_kmb))

    # Set x-axis
    ax1.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    # Combine legends - need to manually set the legend with all lines
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]

    # Create legend with all lines explicitly
    legend = ax1.legend(
        lines,
        labels,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.8,
        borderpad=0.4,
        handlelength=1,
        fontsize=FONT_SIZES["legend"],
    )
    plt.setp(legend.get_texts(), fontsize=FONT_SIZES["legend"], color="#333333")

    apply_grid_style(ax1, major_alpha=0.6, minor_alpha=0.4)

    # Disable grid on ax2 to avoid overlapping
    ax2.grid(False)

    return fig


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    # Create cumulative statistics trend
    fig1 = create_cumulative_stats_trend(df, "author_date")
    fig1.savefig(outputDir / "cumulative_stats_trend.png", dpi=300, bbox_inches="tight")
    fig1.savefig(
        figuresDir / "rq1_cumulative_stats.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig1)

    # Create monthly statistics trend
    fig2 = create_daily_stats_trend(df, "author_date")
    fig2.savefig(outputDir / "monthly_stats_trend.png", dpi=300, bbox_inches="tight")
    fig2.savefig(
        figuresDir / "rq1_monthly_stats.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig2)

    print(f"\nPNG files saved in: {outputDir}")
    print(f"PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
