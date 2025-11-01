import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    FIG_SIZE_MEDIUM,
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


def create_churn_boxplots(df):
    """Create box plots showing distribution of code churn metrics per commit by repository."""
    setup_plotting_style()

    repos = sorted(df["repo_full_name"].unique())
    repo_colors = get_repo_color_mapping(repos)

    # Create three subplots for files changed, insertions, and deletions
    fig, axes = plt.subplots(
        3, 1, figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_MEDIUM[1] * 1.5)
    )

    metrics = [
        ("files_changed", "Files Changed"),
        ("insertions", "Lines Added"),
        ("deletions", "Lines Deleted"),
    ]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for each repository
        data_to_plot = []
        labels = []
        colors = []

        for repo in repos:
            repo_df = df[df["repo_full_name"] == repo].copy()
            # Filter out zeros for insertions and deletions to show meaningful distributions
            if metric in ["insertions", "deletions"]:
                metric_data = repo_df[repo_df[metric] > 0][metric].values
            else:
                metric_data = repo_df[metric].values

            if len(metric_data) > 0:
                data_to_plot.append(metric_data)
                labels.append(REPO_NAME_TO_DISPLAY.get(repo, repo))
                colors.append(repo_colors[repo])

        # Create box plot
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            widths=0.6,
            showfliers=False,  # Don't show outliers for cleaner visualization
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"])
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_SIZES["tick"])
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])

        apply_grid_style(ax, major_alpha=0.3, minor_alpha=0.15)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    df = load_data()

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    print("Creating churn box plots...")
    fig = create_churn_boxplots(df)
    fig.savefig(outputDir / "churn_boxplots.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figuresDir / "rq1_churn_boxplots.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig)

    print(f"\nBox plot PNG file saved in: {outputDir}")
    print(f"Box plot PDF file saved in: {figuresDir}")


if __name__ == "__main__":
    main()
