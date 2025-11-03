import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import numpy as np

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    apply_grid_style,
    FONT_SIZES,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def load_data():
    """Load issues data from parquet file."""
    dataPath = DATA_DIR / "issues.parquet"
    df = pd.read_parquet(dataPath)
    return df


def create_issue_heatmap(df, date_field, period="M"):
    """
    Create a heatmap showing issue creation intensity across repositories and time periods.

    Args:
        df: DataFrame with issue data
        date_field: Column name for the date field to use
        period: Time period for aggregation ('M' for month, 'Q' for quarter, 'Y' for year)
    """
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    # Create time period column
    period_labels = {"M": "Month", "Q": "Quarter", "Y": "Year"}

    df["time_period"] = df[date_field].dt.to_period(period)

    # Group by repository and time period
    heatmap_data = (
        df.groupby(["repo", "time_period"]).size().reset_index(name="issue_count")
    )

    # Pivot to create matrix format
    pivot_data = heatmap_data.pivot(
        index="repo", columns="time_period", values="issue_count"
    )
    pivot_data = pivot_data.fillna(0)

    # Apply display names to repositories
    pivot_data.index = [
        REPO_NAME_TO_DISPLAY.get(repo, repo) for repo in pivot_data.index
    ]

    # Sort by total issues (descending)
    pivot_data["total"] = pivot_data.sum(axis=1)
    pivot_data = pivot_data.sort_values("total", ascending=True)
    pivot_data = pivot_data.drop("total", axis=1)

    # Store original period data for tick selection
    original_columns = pivot_data.columns.copy()

    # Format column labels and identify year marker positions
    if period == "M":
        # For monthly, we'll use custom tick labels later
        pivot_data.columns = [col.strftime("%Y-%m") for col in pivot_data.columns]
        # Identify January months for tick labels (show only year)
        year_marker_indices = [
            i for i, col in enumerate(original_columns) if col.month == 1
        ]
        year_marker_labels = [
            str(original_columns[i].year) for i in year_marker_indices
        ]
    elif period == "Q":
        # For quarterly, keep Q1-Q4 but show year markers at Q1
        pivot_data.columns = [f"{col.quarter}" for col in pivot_data.columns]
        # Identify Q1 (first quarter) for year markers
        year_marker_indices = [
            i for i, col in enumerate(original_columns) if col.quarter == 1
        ]
        year_marker_labels = [
            str(original_columns[i].year) for i in year_marker_indices
        ]
    else:  # Year
        # For yearly, show every year
        pivot_data.columns = [str(col.year) for col in pivot_data.columns]
        # Show tick for every year
        year_marker_indices = list(range(len(pivot_data.columns)))
        year_marker_labels = list(pivot_data.columns)

    # Create figure with single column size and reduced height
    fig, ax = plt.subplots(figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_SINGLE_COL[1]))

    # Create heatmap using a colormap that's suitable for intensity
    # For monthly and quarterly, hide all tick labels initially
    if period in ["M", "Q"]:
        xticklabels_param = False
    else:
        xticklabels_param = pivot_data.columns

    sns.heatmap(
        pivot_data,
        cmap="viridis",
        cbar_kws={
            "label": "Number of Issues",
        },
        linewidths=0,
        ax=ax,
        fmt="g",
        square=False,
        xticklabels=xticklabels_param,
        yticklabels=pivot_data.index,
    )

    # Remove axis labels (self-explanatory)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Show year marker ticks for all period types
    if year_marker_indices:
        # Set tick positions at year markers
        ax.set_xticks([i + 0.5 for i in year_marker_indices])
        ax.set_xticklabels(
            year_marker_labels, rotation=0, ha="center", fontsize=FONT_SIZES["tick"]
        )
        # Show tick marks on x-axis for year marker positions, hide on y-axis
        ax.tick_params(axis="x", which="both", length=5, direction="out")
        ax.tick_params(axis="y", which="both", length=0)
    else:
        # Fallback: remove tick marks on both axes
        ax.tick_params(axis="both", which="both", length=0)
        # Rotate x-axis labels for readability
        plt.setp(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONT_SIZES["tick"]
        )

    # Set y-axis labels with no rotation
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZES["tick"])

    # Adjust colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick"])
    cbar.set_label("Number of Issues", fontsize=FONT_SIZES["legend"])

    plt.tight_layout()

    return fig


def create_issue_heatmap_quarterly(df, date_field):
    """
    Create a quarterly heatmap showing issue creation intensity.
    """
    return create_issue_heatmap(df, date_field, period="Q")


def create_issue_heatmap_monthly(df, date_field):
    """
    Create a monthly heatmap showing issue creation intensity.
    """
    return create_issue_heatmap(df, date_field, period="M")


def create_issue_heatmap_yearly(df, date_field):
    """
    Create a yearly heatmap showing issue creation intensity.
    """
    return create_issue_heatmap(df, date_field, period="Y")


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    # Create quarterly heatmap
    fig1 = create_issue_heatmap_quarterly(df, "issue_created_at")
    fig1.savefig(
        outputDir / "issue_heatmap_quarterly.png", dpi=300, bbox_inches="tight"
    )
    fig1.savefig(
        figuresDir / "rq2_issue_heatmap_quarterly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig1)

    # Create monthly heatmap (might be dense depending on date range)
    fig2 = create_issue_heatmap_monthly(df, "issue_created_at")
    fig2.savefig(outputDir / "issue_heatmap_monthly.png", dpi=300, bbox_inches="tight")
    fig2.savefig(
        figuresDir / "rq2_issue_heatmap_monthly.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig2)

    # Create yearly heatmap
    fig3 = create_issue_heatmap_yearly(df, "issue_created_at")
    fig3.savefig(outputDir / "issue_heatmap_yearly.png", dpi=300, bbox_inches="tight")
    fig3.savefig(
        figuresDir / "rq2_issue_heatmap_yearly.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig3)

    print(f"\nHeatmap PNG files saved in: {outputDir}")
    print(f"Heatmap PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
