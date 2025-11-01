import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from plot_utils import (
    MAIN_COLORS,
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
)
from constants import DATA_DIR


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    # Filter to only include data up to 2025-07-31
    df = df[df["author_date"] <= "2025-07-31"]
    return df


def create_stacked_area_chart(df, date_field):
    """Create stacked area chart showing total lines added vs. deleted over time."""
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    # Group by month and sum across all repos
    df["year_month"] = df[date_field].dt.to_period("M")
    monthly = (
        df.groupby("year_month")
        .agg({"insertions": "sum", "deletions": "sum"})
        .reset_index()
    )

    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.sort_values("date")

    # Create stacked area chart
    ax.fill_between(
        monthly["date"],
        0,
        monthly["insertions"],
        label="Lines Added",
        # color="#2ecc71",
        color=MAIN_COLORS[2],
        alpha=0.7,
    )
    ax.fill_between(
        monthly["date"],
        0,
        -monthly["deletions"],  # Negative for deletions
        label="Lines Deleted",
        # color="#e74c3c",
        color=MAIN_COLORS[3],
        alpha=0.7,
    )

    ax.set_ylabel("Lines Changed", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    # Format y-axis with K/M notation
    def format_thousands(x, pos):
        if abs(x) >= 1e6:
            return f"{x/1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:.0f}K"
        else:
            return f"{x:.0f}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))

    setup_legend(ax, loc="upper left", max_width=15)
    apply_grid_style(ax, major_alpha=0.4, minor_alpha=0.2)

    plt.tight_layout()
    return fig


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    print("Creating stacked area chart...")
    fig = create_stacked_area_chart(df, "author_date")
    fig.savefig(outputDir / "stacked_area_chart.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figuresDir / "rq1_stacked_area_chart.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig)

    print(f"\nStacked area chart PNG file saved in: {outputDir}")
    print(f"Stacked area chart PDF file saved in: {figuresDir}")


if __name__ == "__main__":
    main()
