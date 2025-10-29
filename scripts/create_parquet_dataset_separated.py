"""
Convert all comments.jsonl files into separate Parquet datasets for issues and comments.
"""

import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_repo_name_from_dir(dir_name):
    return dir_name.replace("+", "/")


def extract_issue_and_comments(issue_data, repo_name):
    issue_row = {
        "repo": repo_name,
        "issue_number": issue_data.get("number"),
        "issue_title": issue_data.get("title"),
        "issue_body": issue_data.get("body", ""),
        "issue_created_at": issue_data.get("createdAt"),
        "issue_updated_at": issue_data.get("updatedAt"),
        "issue_closed": issue_data.get("closed", False),
        "issue_closed_at": issue_data.get("closedAt"),
        "issue_author": (
            issue_data.get("author", {}).get("login")
            if issue_data.get("author")
            else None
        ),
        "labels": [
            label["name"] for label in issue_data.get("labels", {}).get("nodes", [])
        ],
        "labels_count": len(issue_data.get("labels", {}).get("nodes", [])),
        "reaction_thumbs_up": 0,
        "reaction_thumbs_down": 0,
        "reaction_laugh": 0,
        "reaction_hooray": 0,
        "reaction_confused": 0,
        "reaction_heart": 0,
        "reaction_rocket": 0,
        "reaction_eyes": 0,
        "total_reactions": 0,
        "comments_count": len(issue_data.get("comments", [])),
        "closing_prs": [
            {"number": pr["number"], "title": pr["title"]}
            for pr in issue_data.get("closedByPullRequests", [])
        ],
        "closing_prs_count": len(issue_data.get("closedByPullRequests", [])),
        "closing_pr_numbers": [
            pr["number"] for pr in issue_data.get("closedByPullRequests", [])
        ],
    }

    reaction_groups = issue_data.get("reactionGroups", [])
    for reaction_group in reaction_groups:
        content = reaction_group.get("content", "").lower()
        count = reaction_group.get("reactors", {}).get("totalCount", 0)
        if content == "thumbs_up":
            issue_row["reaction_thumbs_up"] = count
        elif content == "thumbs_down":
            issue_row["reaction_thumbs_down"] = count
        elif content == "laugh":
            issue_row["reaction_laugh"] = count
        elif content == "hooray":
            issue_row["reaction_hooray"] = count
        elif content == "confused":
            issue_row["reaction_confused"] = count
        elif content == "heart":
            issue_row["reaction_heart"] = count
        elif content == "rocket":
            issue_row["reaction_rocket"] = count
        elif content == "eyes":
            issue_row["reaction_eyes"] = count

    issue_row["total_reactions"] = sum(
        [
            issue_row["reaction_thumbs_up"],
            issue_row["reaction_thumbs_down"],
            issue_row["reaction_laugh"],
            issue_row["reaction_hooray"],
            issue_row["reaction_confused"],
            issue_row["reaction_heart"],
            issue_row["reaction_rocket"],
            issue_row["reaction_eyes"],
        ]
    )

    comments_rows = []
    for comment in issue_data.get("comments", []):
        comment_row = {
            "repo": repo_name,
            "issue_number": issue_data.get("number"),  # Foreign key to issues table
            "comment_id": comment.get("id"),
            "comment_body": comment.get("body", ""),
            "comment_created_at": comment.get("createdAt"),
            "comment_updated_at": comment.get("updatedAt"),
            "comment_author": (
                comment.get("author", {}).get("login")
                if comment.get("author")
                else None
            ),
            "reaction_thumbs_up": 0,
            "reaction_thumbs_down": 0,
            "reaction_laugh": 0,
            "reaction_hooray": 0,
            "reaction_confused": 0,
            "reaction_heart": 0,
            "reaction_rocket": 0,
            "reaction_eyes": 0,
            "total_reactions": 0,
        }

        for reaction_group in comment.get("reactionGroups", []):
            content = reaction_group.get("content", "").lower()
            count = reaction_group.get("reactors", {}).get("totalCount", 0)
            if content == "thumbs_up":
                comment_row["reaction_thumbs_up"] = count
            elif content == "thumbs_down":
                comment_row["reaction_thumbs_down"] = count
            elif content == "laugh":
                comment_row["reaction_laugh"] = count
            elif content == "hooray":
                comment_row["reaction_hooray"] = count
            elif content == "confused":
                comment_row["reaction_confused"] = count
            elif content == "heart":
                comment_row["reaction_heart"] = count
            elif content == "rocket":
                comment_row["reaction_rocket"] = count
            elif content == "eyes":
                comment_row["reaction_eyes"] = count

        comment_row["total_reactions"] = sum(
            [
                comment_row["reaction_thumbs_up"],
                comment_row["reaction_thumbs_down"],
                comment_row["reaction_laugh"],
                comment_row["reaction_hooray"],
                comment_row["reaction_confused"],
                comment_row["reaction_heart"],
                comment_row["reaction_rocket"],
                comment_row["reaction_eyes"],
            ]
        )

        comments_rows.append(comment_row)

    return issue_row, comments_rows


def process_all_repositories_separated(data_dir):
    all_issues = []
    all_comments = []

    repo_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    print(f"Found {len(repo_dirs)} repository directories")

    for repo_dir in repo_dirs:
        repo_name = get_repo_name_from_dir(repo_dir.name)
        comments_file = repo_dir / "comments.jsonl"

        if not comments_file.exists():
            print(f"Warning: No comments.jsonl found for {repo_name}")
            continue

        print(f"Processing {repo_name}...")

        try:
            with open(comments_file, "r", encoding="utf-8") as f:
                issue_count = 0
                comment_count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        issue_data = json.loads(line)
                        issue_row, comments_rows = extract_issue_and_comments(
                            issue_data, repo_name
                        )
                        all_issues.append(issue_row)
                        all_comments.extend(comments_rows)
                        issue_count += 1
                        comment_count += len(comments_rows)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {repo_name}: {e}")
                        continue

                print(
                    f"  Processed {issue_count} issues, {comment_count} comments from {repo_name}"
                )

        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
            continue

    if not all_issues:
        print("No data processed from any repository")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\nTotal issues created: {len(all_issues)}")
    print(f"Total comments created: {len(all_comments)}")

    issues_df = pd.DataFrame(all_issues)
    comments_df = pd.DataFrame(all_comments) if all_comments else pd.DataFrame()

    datetime_columns = ["issue_created_at", "issue_updated_at", "issue_closed_at"]

    for col in datetime_columns:
        if col in issues_df.columns:
            issues_df[col] = pd.to_datetime(issues_df[col], errors="coerce")

    if not comments_df.empty:
        comment_datetime_columns = ["comment_created_at", "comment_updated_at"]

        for col in comment_datetime_columns:
            if col in comments_df.columns:
                comments_df[col] = pd.to_datetime(comments_df[col], errors="coerce")

    return issues_df, comments_df


def create_parquet_datasets_separated():
    """Main function to create separated Parquet datasets."""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    issues_parquet_file = output_dir / "issues.parquet"
    comments_parquet_file = output_dir / "comments.parquet"

    print("Starting separated Parquet datasets creation...")
    print(f"Data directory: {data_dir}")
    print(f"Issues output file: {issues_parquet_file}")
    print(f"Comments output file: {comments_parquet_file}")

    issues_df, comments_df = process_all_repositories_separated(data_dir)

    if issues_df.empty:
        print("Error: No data was processed. Please check your data directory.")
        return

    print(f"\nDataset Statistics:")
    print(f"Total issues: {len(issues_df):,}")
    print(f"Total comments: {len(comments_df):,}")
    print(f"Total repositories: {issues_df['repo'].nunique()}")

    print(f"\nRepository breakdown:")
    repo_stats_issues = (
        issues_df.groupby("repo")
        .agg({"issue_number": "count"})
        .rename(columns={"issue_number": "issues"})
    )

    if not comments_df.empty:
        repo_stats_comments = (
            comments_df.groupby("repo")
            .agg({"comment_id": "count"})
            .rename(columns={"comment_id": "comments"})
        )

        repo_stats = repo_stats_issues.join(repo_stats_comments, how="left").fillna(0)
        repo_stats["comments"] = repo_stats["comments"].astype(int)
    else:
        repo_stats = repo_stats_issues
        repo_stats["comments"] = 0

    for repo, stats in repo_stats.iterrows():
        print(f"  {repo}: {stats['issues']} issues, {stats['comments']} comments")

    print(f"\nSaving to Parquet format...")

    try:
        issues_table = pa.Table.from_pandas(issues_df)
        pq.write_table(
            issues_table,
            issues_parquet_file,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )

        issues_file_size = issues_parquet_file.stat().st_size
        issues_file_size_mb = issues_file_size / (1024 * 1024)

        print(f"Successfully created issues Parquet dataset!")
        print(f"Issues file: {issues_parquet_file}")
        print(f"Issues size: {issues_file_size_mb:.2f} MB")

        if not comments_df.empty:
            comments_table = pa.Table.from_pandas(comments_df)
            pq.write_table(
                comments_table,
                comments_parquet_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
            )

            comments_file_size = comments_parquet_file.stat().st_size
            comments_file_size_mb = comments_file_size / (1024 * 1024)

            print(f"Successfully created comments Parquet dataset!")
            print(f"Comments file: {comments_parquet_file}")
            print(f"Comments size: {comments_file_size_mb:.2f} MB")
        else:
            print("No comments found, skipping comments table creation.")

        print(f"\nVerifying Parquet files...")

        test_issues_table = pq.read_table(issues_parquet_file)
        test_issues_df = test_issues_table.to_pandas()
        print(f"Issues verification successful: {len(test_issues_df):,} issues read")

        if not comments_df.empty:
            test_comments_table = pq.read_table(comments_parquet_file)
            test_comments_df = test_comments_table.to_pandas()
            print(
                f"Comments verification successful: {len(test_comments_df):,} comments read"
            )

    except Exception as e:
        print(f"Error saving Parquet files: {e}")
        raise


if __name__ == "__main__":
    create_parquet_datasets_separated()
