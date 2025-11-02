import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
    PLOT_LINE_WIDTH,
    FIG_SIZE_MEDIUM,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY
import seaborn as sns


def load_data():
    """Load and filter data for the three target repositories."""
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    # Filter to only include data up to 2025-07-31
    df = df[df["author_date"] <= "2025-07-31"]

    # Filter for the three repositories of interest
    target_repos = [
        "langchain-ai/langchain",
        "deepset-ai/haystack",
        "TransformerOptimus/SuperAGI",
    ]
    df = df[df["repo_full_name"].isin(target_repos)].copy()

    return df


def create_monthly_aggregation(df):
    """Aggregate code churn metrics by month for each repository."""
    df = df.copy()
    if df["author_date"].dt.tz is not None:
        df["author_date"] = df["author_date"].dt.tz_convert(None)

    df["year_month"] = df["author_date"].dt.to_period("M")

    monthly = (
        df.groupby(["repo_full_name", "year_month"])
        .agg(
            {
                "insertions": "sum",
                "deletions": "sum",
                "files_changed": "sum",
            }
        )
        .reset_index()
    )

    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.sort_values("date")

    return monthly


def create_code_churn_heatmap(df):
    """Create heatmap showing code churn intensity across repos and time."""
    setup_plotting_style()

    monthly = create_monthly_aggregation(df)

    # Create figure with 3 heatmaps (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

    repos = sorted(monthly["repo_full_name"].unique())
    metrics = [
        ("insertions", "Lines Added"),
        ("deletions", "Lines Deleted"),
        ("files_changed", "Files Changed"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Pivot data for heatmap
        pivot_data = monthly.pivot(
            index="repo_full_name", columns="date", values=metric
        ).fillna(0)

        # Reorder repos and rename
        pivot_data = pivot_data.reindex(repos)
        pivot_data.index = [REPO_NAME_TO_DISPLAY.get(r, r) for r in pivot_data.index]

        # Select subset of columns for readability (yearly samples)
        date_cols = pivot_data.columns
        # Sample every 12 months
        sample_indices = range(0, len(date_cols), 12)
        sampled_data = pivot_data.iloc[:, sample_indices]

        # Format column labels as years
        sampled_data.columns = [d.strftime("%Y") for d in sampled_data.columns]

        # Create heatmap
        sns.heatmap(
            sampled_data,
            ax=ax,
            cmap="YlOrRd",
            cbar_kws={"label": title},
            linewidths=0.5,
            linecolor="#EEEEEE",
            fmt=".0f",
            square=False,
            xticklabels=True,
            yticklabels=True if idx == 0 else False,
        )

        ax.set_xlabel("Year", fontsize=FONT_SIZES["axis_label"])
        if idx == 0:
            ax.set_ylabel("Repository", fontsize=FONT_SIZES["axis_label"])
        else:
            ax.set_ylabel("")

        ax.set_title(title, fontsize=FONT_SIZES["axis_label"], pad=8)
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        if idx == 0:
            plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig


def format_kmb(x, pos):
    """Format y-axis labels with K/M/B notation without unnecessary decimals."""
    if abs(x) >= 1e9:
        val = x / 1e9
        return (
            f"{val:.1f}B".rstrip("0").rstrip(".") if val != int(val) else f"{int(val)}B"
        )
    elif abs(x) >= 1e6:
        val = x / 1e6
        return (
            f"{val:.1f}M".rstrip("0").rstrip(".") if val != int(val) else f"{int(val)}M"
        )
    elif abs(x) >= 1e3:
        val = x / 1e3
        return (
            f"{val:.1f}K".rstrip("0").rstrip(".") if val != int(val) else f"{int(val)}K"
        )
    else:
        return f"{int(x)}" if x == int(x) else f"{x:.1f}".rstrip("0").rstrip(".")


def create_code_churn_faceted(df):
    """Create faceted area charts showing churn over time for each repo."""
    setup_plotting_style()

    monthly = create_monthly_aggregation(df)

    repos = sorted(monthly["repo_full_name"].unique())

    # Define consistent colors for metrics (not repo-specific)
    color_insertions = "#2ecc71"  # Green for additions
    color_deletions = "#e74c3c"  # Red for deletions
    color_files = "#3498db"  # Blue for files

    # Create 3 subplots (one per repository)
    fig, axes = plt.subplots(3, 1, figsize=(FIG_SIZE_SINGLE_COL[0], 4.5), sharex=False)

    for idx, repo in enumerate(repos):
        ax = axes[idx]
        repo_data = monthly[monthly["repo_full_name"] == repo].sort_values("date")

        # Compute date range for this specific repository with padding
        repo_min = repo_data["date"].min()
        repo_max = repo_data["date"].max()
        date_range = repo_max - repo_min
        padding = date_range * 0.02  # 2% padding on each side
        repo_min_padded = repo_min - padding
        repo_max_padded = repo_max + padding

        # Plot all three metrics with consistent colors
        ax.fill_between(
            repo_data["date"],
            0,
            repo_data["insertions"],
            label="Lines Added",
            color=color_insertions,
            alpha=0.7,
        )

        ax.fill_between(
            repo_data["date"],
            0,
            repo_data["deletions"],
            label="Lines Deleted",
            color=color_deletions,
            alpha=0.5,
        )

        # Plot files changed as a line overlay
        ax2 = ax.twinx()
        ax2.plot(
            repo_data["date"],
            repo_data["files_changed"],
            label="Files Changed",
            color=color_files,
            linewidth=PLOT_LINE_WIDTH,
            linestyle="-",
            alpha=0.8,
        )

        # Configure primary axis (lines)
        ax.set_ylabel(
            f"{REPO_NAME_TO_DISPLAY.get(repo, repo)}\nLines",
            fontsize=FONT_SIZES["axis_label"],
        )
        # Apply repository-specific x-limits with padding
        ax.set_xlim(repo_min_padded, repo_max_padded)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_kmb))
        apply_grid_style(ax, major_alpha=0.4, minor_alpha=0.2)

        # Configure secondary axis (files)
        ax2.set_ylabel("Files", fontsize=FONT_SIZES["tick"], rotation=270, labelpad=12)
        ax2.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_kmb))
        # Disable grid on secondary axis to avoid overlapping gridlines
        ax2.grid(False)

        # Configure x-axis with repository-specific date range
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Only show legend on first plot; create a consistent legend order
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Keep legend compact and consistent across charts
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize=FONT_SIZES["legend"],
                frameon=True,
                facecolor="white",
                edgecolor="#CCCCCC",
                framealpha=0.8,
            )

    plt.xticks(rotation=0)
    plt.tight_layout()

    return fig


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    print("Creating code churn comparison chart...")

    # Create faceted area chart
    fig = create_code_churn_faceted(df)
    fig.savefig(outputDir / "code_churn_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figuresDir / "rq1_code_churn_comparison.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig)

    print(f"\nCode churn comparison PNG file saved in: {outputDir}")
    print(f"Code churn comparison PDF file saved in: {figuresDir}")


if __name__ == "__main__":
    main()
