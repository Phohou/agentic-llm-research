import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
    MARKER_SIZE,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    # Filter to only include data up to 2025-07-31
    df = df[df["author_date"] <= "2025-07-31"]
    return df


def create_commits_vs_files_scatter(df):
    """Create scatter plot of commits vs. files changed."""
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    repos = sorted(df["repo_full_name"].unique())
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()
        repo_df = repo_df.sort_values("author_date")

        # Use commit index
        x = range(1, len(repo_df) + 1)
        y = repo_df["files_changed"].values

        ax.scatter(
            x,
            y,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            alpha=0.5,
            s=MARKER_SIZE,
            edgecolors="none",
        )

    ax.set_xlabel("Commit Number", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Files Changed per Commit", fontsize=FONT_SIZES["axis_label"])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)

    setup_legend(ax, loc="upper left", max_width=10)
    apply_grid_style(ax, major_alpha=0.4, minor_alpha=0.2)

    plt.tight_layout()
    return fig


def create_commits_vs_insertions_scatter(df):
    """Create scatter plot of commits vs. lines added."""
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    repos = sorted(df["repo_full_name"].unique())
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()
        repo_df = repo_df.sort_values("author_date")

        # Filter out commits with 0 insertions for log scale
        repo_df = repo_df[repo_df["insertions"] > 0]

        x = range(1, len(repo_df) + 1)
        y = repo_df["insertions"].values

        ax.scatter(
            x,
            y,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            alpha=0.5,
            s=MARKER_SIZE,
            edgecolors="none",
        )

    ax.set_xlabel("Commit Number", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Lines Added per Commit", fontsize=FONT_SIZES["axis_label"])
    ax.set_yscale("log")

    setup_legend(ax, loc="upper left", max_width=10)
    apply_grid_style(ax, major_alpha=0.4, minor_alpha=0.2)

    plt.tight_layout()
    return fig


def create_commits_vs_deletions_scatter(df):
    """Create scatter plot of commits vs. lines deleted."""
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    repos = sorted(df["repo_full_name"].unique())
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_full_name"] == repo].copy()
        repo_df = repo_df.sort_values("author_date")

        # Filter out commits with 0 deletions for log scale
        repo_df = repo_df[repo_df["deletions"] > 0]

        x = range(1, len(repo_df) + 1)
        y = repo_df["deletions"].values

        ax.scatter(
            x,
            y,
            label=REPO_NAME_TO_DISPLAY.get(repo, repo),
            color=repo_colors[repo],
            alpha=0.5,
            s=MARKER_SIZE,
            edgecolors="none",
        )

    ax.set_xlabel("Commit Number", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Lines Deleted per Commit", fontsize=FONT_SIZES["axis_label"])
    ax.set_yscale("log")

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

    print("Creating scatter plots...")

    fig1 = create_commits_vs_files_scatter(df)
    fig1.savefig(
        outputDir / "commits_vs_files_scatter.png", dpi=300, bbox_inches="tight"
    )
    fig1.savefig(
        figuresDir / "rq1_commits_vs_files_scatter.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig1)

    fig2 = create_commits_vs_insertions_scatter(df)
    fig2.savefig(
        outputDir / "commits_vs_insertions_scatter.png", dpi=300, bbox_inches="tight"
    )
    fig2.savefig(
        figuresDir / "rq1_commits_vs_insertions_scatter.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig2)

    fig3 = create_commits_vs_deletions_scatter(df)
    fig3.savefig(
        outputDir / "commits_vs_deletions_scatter.png", dpi=300, bbox_inches="tight"
    )
    fig3.savefig(
        figuresDir / "rq1_commits_vs_deletions_scatter.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close(fig3)

    print(f"\nScatter plot PNG files saved in: {outputDir}")
    print(f"Scatter plot PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
