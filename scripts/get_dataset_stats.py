"""
Generate dataset statistics for issues and commits per repository.

This script analyzes:
- Commits: Total count and deduplicated count
- Issues: Total, with PRs, with labels, and with both
"""

import pandas as pd
from pathlib import Path


def generate_dataset_statistics():
    """Generate comprehensive dataset statistics."""

    # Define file paths
    base_path = Path(__file__).parent.parent / "data"
    issues_path = base_path / "issues.parquet"
    commits_path = base_path / "combined_commits.parquet"
    commits_deduped_path = base_path / "combined_commits_deduped.parquet"

    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    # ========================
    # COMMITS STATISTICS
    # ========================
    print("\n📊 COMMITS ANALYSIS")
    print("-" * 80)

    # Load commits data
    commits_df = pd.read_parquet(commits_path)
    commits_deduped_df = pd.read_parquet(commits_deduped_path)

    # Count by repository
    commits_by_repo = (
        commits_df.groupby("repo_full_name").size().reset_index(name="total_commits")
    )
    commits_deduped_by_repo = (
        commits_deduped_df.groupby("repo_full_name")
        .size()
        .reset_index(name="deduped_commits")
    )

    # Merge commits stats
    commits_stats = pd.merge(
        commits_by_repo, commits_deduped_by_repo, on="repo_full_name", how="outer"
    )
    commits_stats = commits_stats.fillna(0).astype(
        {"total_commits": int, "deduped_commits": int}
    )
    commits_stats = commits_stats.sort_values("repo_full_name")

    print("\nCommits by Repository:")
    print(commits_stats.to_string(index=False))

    total_commits = commits_df.shape[0]
    total_deduped_commits = commits_deduped_df.shape[0]
    print(f"\n{'Total':<40} {total_commits:>15,} {total_deduped_commits:>15,}")

    # ========================
    # ISSUES STATISTICS
    # ========================
    print("\n\n📊 ISSUES ANALYSIS")
    print("-" * 80)

    # Load issues data
    issues_df = pd.read_parquet(issues_path)

    # Group by issue_number and repo to get unique issues (since comments create multiple rows)
    unique_issues = (
        issues_df.groupby(["repo", "issue_number"])
        .agg({"closing_prs_count": "first", "labels_count": "first"})
        .reset_index()
    )

    # Calculate statistics per repository
    issue_stats = []

    for repo in sorted(unique_issues["repo"].unique()):
        repo_issues = unique_issues[unique_issues["repo"] == repo]

        total = len(repo_issues)
        with_prs = len(repo_issues[repo_issues["closing_prs_count"] > 0])
        with_labels = len(repo_issues[repo_issues["labels_count"] > 0])
        with_both = len(
            repo_issues[
                (repo_issues["closing_prs_count"] > 0)
                & (repo_issues["labels_count"] > 0)
            ]
        )

        issue_stats.append(
            {
                "Repo": repo.split("/")[-1],  # Just the repo name without owner
                "Total": total,
                "w/ PRs": with_prs,
                "w/ Labels": with_labels,
                "w/ Both": with_both,
            }
        )

    # Calculate totals
    total_issues = len(unique_issues)
    total_with_prs = len(unique_issues[unique_issues["closing_prs_count"] > 0])
    total_with_labels = len(unique_issues[unique_issues["labels_count"] > 0])
    total_with_both = len(
        unique_issues[
            (unique_issues["closing_prs_count"] > 0)
            & (unique_issues["labels_count"] > 0)
        ]
    )

    # Create DataFrame for display
    issues_stats_df = pd.DataFrame(issue_stats)

    print("\nIssues by Repository:")
    print(
        f"{'Repo':<20} {'Total':>10} {'w/ PRs':>10} {'w/ Labels':>12} {'w/ Both':>10}"
    )
    print("-" * 80)
    for _, row in issues_stats_df.iterrows():
        print(
            f"{row['Repo']:<20} {row['Total']:>10,} {row['w/ PRs']:>10,} {row['w/ Labels']:>12,} {row['w/ Both']:>10,}"
        )

    print("-" * 80)
    print(
        f"{'Total':<20} {total_issues:>10,} {total_with_prs:>10,} {total_with_labels:>12,} {total_with_both:>10,}"
    )

    # ========================
    # COMBINED SUMMARY TABLE
    # ========================
    print("\n\n📊 COMBINED SUMMARY")
    print("=" * 80)

    # Map repo names for alignment
    repo_mapping = {
        "TransformerOptimus/SuperAGI": "SuperAGI",
        "crewAIInc/crewAI": "CrewAI",
        "deepset-ai/haystack": "Haystack",
        "langchain-ai/langchain": "LangChain",
        "letta-ai/letta": "Letta",
        "microsoft/autogen": "AutoGen",
        "microsoft/semantic-kernel": "Semantic_Kernel",
        "run-llama/llama_index": "LlamaIndex",
    }

    # Create combined table
    combined_stats = []

    for full_repo_name in sorted(repo_mapping.keys()):
        short_name = repo_mapping[full_repo_name]

        # Get commits stats
        commit_row = commits_stats[commits_stats["repo_full_name"] == full_repo_name]
        if not commit_row.empty:
            total_commits_repo = commit_row["total_commits"].values[0]
            deduped_commits_repo = commit_row["deduped_commits"].values[0]
        else:
            total_commits_repo = 0
            deduped_commits_repo = 0

        # Get issues stats
        repo_issues = unique_issues[unique_issues["repo"] == full_repo_name]
        total_issues_repo = len(repo_issues)
        with_prs_repo = len(repo_issues[repo_issues["closing_prs_count"] > 0])
        with_labels_repo = len(repo_issues[repo_issues["labels_count"] > 0])
        with_both_repo = len(
            repo_issues[
                (repo_issues["closing_prs_count"] > 0)
                & (repo_issues["labels_count"] > 0)
            ]
        )

        combined_stats.append(
            {
                "Repo": short_name,
                "Total Issues": total_issues_repo,
                "w/ PRs": with_prs_repo,
                "w/ Labels": with_labels_repo,
                "w/ Both": with_both_repo,
                "Total Commits": total_commits_repo,
                "Deduped Commits": deduped_commits_repo,
            }
        )

    combined_df = pd.DataFrame(combined_stats)

    # Print header
    print(f"\n{'Repo':<20} | {'Issues':<50} | {'Commits':<30}")
    print(
        f"{'':<20} | {'Total':>10} {'w/ PRs':>10} {'w/ Labels':>12} {'w/ Both':>10} | {'Total':>12} {'Deduped':>12}"
    )
    print("-" * 115)

    # Print each row
    for _, row in combined_df.iterrows():
        print(
            f"{row['Repo']:<20} | {row['Total Issues']:>10,} {row['w/ PRs']:>10,} {row['w/ Labels']:>12,} {row['w/ Both']:>10,} | {row['Total Commits']:>12,} {row['Deduped Commits']:>12,}"
        )

    # Print totals
    print("-" * 115)
    print(
        f"{'Total':<20} | {total_issues:>10,} {total_with_prs:>10,} {total_with_labels:>12,} {total_with_both:>10,} | {total_commits:>12,} {total_deduped_commits:>12,}"
    )
    print("=" * 115)

    # ========================
    # SAVE TO FILE
    # ========================
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)

    # Save detailed CSV
    combined_df.to_csv(output_path / "dataset_statistics.csv", index=False)
    print(f"\n✅ Statistics saved to: {output_path / 'dataset_statistics.csv'}")

    # Save text summary
    with open(output_path / "dataset_statistics.txt", "w") as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 115 + "\n\n")
        f.write(f"{'Repo':<20} | {'Issues':<50} | {'Commits':<30}\n")
        f.write(
            f"{'':<20} | {'Total':>10} {'w/ PRs':>10} {'w/ Labels':>12} {'w/ Both':>10} | {'Total':>12} {'Deduped':>12}\n"
        )
        f.write("-" * 115 + "\n")

        for _, row in combined_df.iterrows():
            f.write(
                f"{row['Repo']:<20} | {row['Total Issues']:>10,} {row['w/ PRs']:>10,} {row['w/ Labels']:>12,} {row['w/ Both']:>10,} | {row['Total Commits']:>12,} {row['Deduped Commits']:>12,}\n"
            )

        f.write("-" * 115 + "\n")
        f.write(
            f"{'Total':<20} | {total_issues:>10,} {total_with_prs:>10,} {total_with_labels:>12,} {total_with_both:>10,} | {total_commits:>12,} {total_deduped_commits:>12,}\n"
        )

    print(f"✅ Text summary saved to: {output_path / 'dataset_statistics.txt'}")

    return combined_df


if __name__ == "__main__":
    generate_dataset_statistics()
