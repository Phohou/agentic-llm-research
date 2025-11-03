import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
    GREY_COLORS_DARK,
    setup_legend,
)
from constants import DATA_DIR, REPO_NAME_TO_DISPLAY


def loadDf():
    """Loads the dataset from the parquet file."""
    dataPath = DATA_DIR / "issues.parquet"
    df = pd.read_parquet(dataPath)
    return df


def load_commits_df():
    """Loads the commits dataset from the parquet file."""
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    return df


def total_issues_repo(df, repo_col):
    """Create a horizontal bar chart showing total issues by repository with cleaned names"""
    setup_plotting_style()

    repo_counts = df[repo_col].value_counts()
    cleaned_labels = [
        REPO_NAME_TO_DISPLAY.get(repo, repo) for repo in repo_counts.index
    ]

    # Get consistent color mapping
    repo_colors = get_repo_color_mapping(repo_counts.index)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    bars = ax.barh(
        cleaned_labels,
        repo_counts.values,
        color=[repo_colors[repo] for repo in repo_counts.index],
    )

    for bar, value in zip(bars, repo_counts.values):
        ax.text(
            value + max(repo_counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            ha="left",
            va="center",
            fontsize=FONT_SIZES["annotation"],
        )

    ax.set_xlabel("Number of Issues", fontsize=FONT_SIZES["axis_label"])

    # Set x-axis limits
    ax.set_xlim(0, 4800)

    apply_grid_style(ax)
    plt.tight_layout()

    return fig


def issues_and_commits_dual_axis(df, commits_df, issues_col, commits_col):
    """Create a horizontal bar chart with dual x-axes showing both issues and commits"""
    setup_plotting_style()

    # Get issues counts
    issues_counts = df[issues_col].value_counts()

    # Get commits counts
    commits_counts = commits_df[commits_col].value_counts()

    # Get all repositories (union of both datasets) and sort by display name (repo name only) in reverse for top-to-bottom display
    all_repos = sorted(
        set(issues_counts.index) | set(commits_counts.index),
        key=lambda x: REPO_NAME_TO_DISPLAY.get(x, x),
        reverse=True,
    )

    # Prepare data
    cleaned_labels = [REPO_NAME_TO_DISPLAY.get(repo, repo) for repo in all_repos]
    issues_values = [issues_counts.get(repo, 0) for repo in all_repos]
    commits_values = [commits_counts.get(repo, 0) for repo in all_repos]

    dark_color = GREY_COLORS_DARK[0]
    light_color = GREY_COLORS_DARK[4]

    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    # Set up positions for bars
    y_pos = range(len(cleaned_labels))
    bar_height = 0.35

    # Calculate the common scale based on the maximum of both datasets
    max_value = max(max(issues_values), max(commits_values))
    common_limit = max_value * 1.15

    # Set up axes properties and grid FIRST before plotting
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cleaned_labels)
    ax1.set_xlim(0, common_limit)

    # Create second x-axis for commits (top)
    ax2 = ax1.twiny()
    ax2.set_xlim(0, common_limit)
    ax2.set_xticklabels([])  # Hide tick labels since both axes share the same scale
    ax2.tick_params(axis="x", which="both", length=0)  # Hide tick marks

    # NOW plot issues on the first axis (bottom)
    bars1 = ax1.barh(
        [y - bar_height / 2 for y in y_pos],
        issues_values,
        bar_height,
        label="Issues",
        color=dark_color,
    )

    # Add value labels for issues (positioned at bottom of bar to avoid overlap)
    for bar, value in zip(bars1, issues_values):
        if value > 0:
            ax1.text(
                value + max_value * 0.01,
                bar.get_y() + bar.get_height() * 0.2,
                f"{int(value)}",
                ha="left",
                va="center",
                fontsize=FONT_SIZES["annotation"] - 1,
            )

    # Plot commits on the second axis (top)
    bars2 = ax2.barh(
        [y + bar_height / 2 for y in y_pos],
        commits_values,
        bar_height,
        label="Commits",
        color=light_color,
    )

    # Add value labels for commits
    for bar, value in zip(bars2, commits_values):
        if value > 0:
            ax2.text(
                value + max_value * 0.01,
                bar.get_y() + bar.get_height() * 0.5,
                f"{int(value)}",
                ha="left",
                va="center",
                fontsize=FONT_SIZES["annotation"] - 1,
            )

    # Apply grid style only to ax1 and ensure it appears behind
    apply_grid_style(ax1)
    # Disable grid on ax2 to prevent gridlines from appearing on top
    ax2.grid(False)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Create legend with combined handles and labels using setup_legend style
    legend = ax1.legend(
        all_handles,
        all_labels,
        loc="lower right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.8,
        borderpad=0.4,
        handlelength=1,
    )
    plt.setp(legend.get_texts(), fontsize=FONT_SIZES["legend"], color="#333333")

    plt.tight_layout()

    return fig


def generate_repository_summary(df, commits_df=None):
    """
    Generate a comprehensive summary table for each repository including:
    - Total number of closed issues
    - Closed issues with linked PRs (closing_prs_count > 0)
    - Closed issues with at least one label (labels_count > 0)
    - Closed issues with both linked PRs and labels
    - Total number of comments across all issues
    - Total number of commits (if commits_df is provided)
    """
    summary_data = []

    # Create commit counts dictionary if commits data is available
    commit_counts = {}
    if commits_df is not None:
        commit_counts = commits_df["repo_full_name"].value_counts().to_dict()

    for repo in sorted(df["repo"].unique()):
        repo_df = df[df["repo"] == repo]

        total_issues = len(repo_df)
        issues_with_prs = len(repo_df[repo_df["closing_prs_count"] > 0])
        issues_with_labels = len(repo_df[repo_df["labels_count"] > 0])
        issues_with_both = len(
            repo_df[(repo_df["closing_prs_count"] > 0) & (repo_df["labels_count"] > 0)]
        )
        total_comments = repo_df["comments_count"].sum()
        total_commits = commit_counts.get(repo, 0)

        # Calculate percentages
        pct_with_prs = (issues_with_prs / total_issues * 100) if total_issues > 0 else 0
        pct_with_labels = (
            (issues_with_labels / total_issues * 100) if total_issues > 0 else 0
        )
        pct_with_both = (
            (issues_with_both / total_issues * 100) if total_issues > 0 else 0
        )
        avg_comments = total_comments / total_issues if total_issues > 0 else 0

        summary_data.append(
            {
                "Repository": REPO_NAME_TO_DISPLAY.get(repo, repo),
                "Issues": total_issues,
                "w/ PRs": issues_with_prs,
                "w/ Labels": issues_with_labels,
                "w/ Both": issues_with_both,
                "Comments": total_comments,
                "Commits": total_commits,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def print_summary_table(summary_df):
    """Print the summary table in a formatted way"""
    print("\n" + "=" * 100)
    print("REPOSITORY SUMMARY - CLOSED ISSUES ANALYSIS")
    print("=" * 100)
    print()

    # Print table with proper formatting and reduced column spacing
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.colheader_justify", "right")
    print(summary_df.to_string(index=False, col_space=8))
    print()
    print("=" * 100)

    # Print overall statistics
    total_issues = summary_df["Issues"].sum()
    total_comments = summary_df["Comments"].sum()
    total_commits = summary_df["Commits"].sum()
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Closed Issues Across All Repositories: {total_issues:,}")
    print(f"  Total Comments Across All Repositories: {total_comments:,}")
    print(f"  Total Commits Across All Repositories: {total_commits:,}")
    print(f"  Average Comments per Issue (Overall): {total_comments/total_issues:.1f}")
    print()


def save_summary_to_files(summary_df, output_dir):
    """Save the summary to CSV and text files"""
    # Save to CSV
    csv_path = output_dir / "repository_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Save to formatted text file
    txt_path = output_dir / "repository_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("REPOSITORY SUMMARY - CLOSED ISSUES ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        f.write(summary_df.to_string(index=False, col_space=8))
        f.write("\n\n" + "=" * 100 + "\n")

        # Add overall statistics
        total_issues = summary_df["Issues"].sum()
        total_comments = summary_df["Comments"].sum()
        total_commits = summary_df["Commits"].sum()
        f.write(f"\nOVERALL STATISTICS:\n")
        f.write(f"  Total Closed Issues Across All Repositories: {total_issues:,}\n")
        f.write(f"  Total Comments Across All Repositories: {total_comments:,}\n")
        f.write(f"  Total Commits Across All Repositories: {total_commits:,}\n")
        f.write(
            f"  Average Comments per Issue (Overall): {total_comments/total_issues:.1f}\n"
        )

    print(f"Summary saved to: {txt_path}")


def main():
    try:
        # Load the dataset
        print("Loading issues dataset...")
        df = loadDf()
        if df is None:
            print("Failed to load issues dataset")
            return

        # Load commits dataset
        print("Loading commits dataset...")
        commits_df = load_commits_df()
        if commits_df is None:
            print(
                "Warning: Failed to load commits dataset. Continuing without commit counts."
            )
            commits_df = None

        # Create output directories
        outputDir = Path("output")
        figuresDir = Path("figures")
        outputDir.mkdir(parents=True, exist_ok=True)
        figuresDir.mkdir(parents=True, exist_ok=True)

        # Generate and display summary table
        print("\nGenerating repository summary...")
        summary_df = generate_repository_summary(df, commits_df)
        print_summary_table(summary_df)

        # Save summary to files
        save_summary_to_files(summary_df, outputDir)

        # Create the bar chart
        print("\nCreating total issues by repository chart...")
        fig = total_issues_repo(df, "repo")
        fig.savefig(
            outputDir / "total_issues_by_repository.png", dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            figuresDir / "total_issues_by_repository.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close(fig)

        print(f"  PNG: {outputDir / 'total_issues_by_repository.png'}")
        print(f"  PDF: {figuresDir / 'total_issues_by_repository.pdf'}")

        # Create the dual-axis chart (issues and commits)
        if commits_df is not None:
            print("\nCreating issues and commits dual-axis chart...")
            fig2 = issues_and_commits_dual_axis(
                df, commits_df, "repo", "repo_full_name"
            )
            fig2.savefig(
                outputDir / "issues_and_commits_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig2.savefig(
                figuresDir / "issues_and_commits_comparison.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close(fig2)

            print(f"  PNG: {outputDir / 'issues_and_commits_comparison.png'}")
            print(f"  PDF: {figuresDir / 'issues_and_commits_comparison.pdf'}")

        print("\nAll files generated successfully!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
