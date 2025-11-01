import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

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
    # Filter to only include data up to 2025-07-31
    df = df[df["author_date"] <= "2025-07-31"]
    return df


def create_add_delete_ratio_chart(df, date_field):
    """Create chart showing lines added/deleted ratio over time."""
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    repos = sorted(df["repo_full_name"].unique())
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()

        # Group by month
        repo_df["year_month"] = repo_df[date_field].dt.to_period("M")
        monthly = (
            repo_df.groupby("year_month")
            .agg({"insertions": "sum", "deletions": "sum"})
            .reset_index()
        )

        monthly["date"] = monthly["year_month"].dt.to_timestamp()
        monthly = monthly.sort_values("date")

        # Calculate ratio (add small epsilon to avoid division by zero)
        monthly["ratio"] = (monthly["insertions"] + 1) / (monthly["deletions"] + 1)

        ax.plot(
            monthly["date"],
            monthly["ratio"],
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH_THIN,
            alpha=0.8,
        )

    ax.set_ylabel("Lines Added/Deleted Ratio", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    setup_legend(ax, loc="upper left", max_width=10)
    apply_grid_style(ax, major_alpha=0.4, minor_alpha=0.2)

    plt.tight_layout()
    return fig


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    print("Creating add/delete ratio chart...")
    fig = create_add_delete_ratio_chart(df, "author_date")
    fig.savefig(outputDir / "add_delete_ratio_chart.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figuresDir / "rq1_add_delete_ratio_chart.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig)

    print(f"\nRatio chart PNG file saved in: {outputDir}")
    print(f"Ratio chart PDF file saved in: {figuresDir}")


if __name__ == "__main__":
    main()
