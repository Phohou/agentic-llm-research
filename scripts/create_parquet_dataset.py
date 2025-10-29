"""
Convert all comments.jsonl files into a single Parquet dataset.
"""

import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_repo_name_from_dir(dir_name):
    return dir_name.replace("+", "/")


def flatten_issue_data(issue_data, repo_name):
    rows = []

    issue_row = {
        "repo": repo_name,
        "type": "issue",
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
        "comment_id": None,
        "comment_body": None,
        "comment_created_at": None,
        "comment_updated_at": None,
        "comment_author": None,
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

    rows.append(issue_row)

    for comment in issue_data.get("comments", []):
        comment_row = issue_row.copy()
        comment_row.update(
            {
                "type": "comment",
                "comment_id": comment.get("id"),
                "comment_body": comment.get("body", ""),
                "comment_created_at": comment.get("createdAt"),
                "comment_updated_at": comment.get("updatedAt"),
                "comment_author": (
                    comment.get("author", {}).get("login")
                    if comment.get("author")
                    else None
                ),
            }
        )

        comment_reactions = {
            "reaction_thumbs_up": 0,
            "reaction_thumbs_down": 0,
            "reaction_laugh": 0,
            "reaction_hooray": 0,
            "reaction_confused": 0,
            "reaction_heart": 0,
            "reaction_rocket": 0,
            "reaction_eyes": 0,
        }

        reaction_groups = comment.get("reactionGroups", [])
        for reaction_group in reaction_groups:
            content = reaction_group.get("content", "").lower()
            count = reaction_group.get("reactors", {}).get("totalCount", 0)
            if content == "thumbs_up":
                comment_reactions["reaction_thumbs_up"] = count
            elif content == "thumbs_down":
                comment_reactions["reaction_thumbs_down"] = count
            elif content == "laugh":
                comment_reactions["reaction_laugh"] = count
            elif content == "hooray":
                comment_reactions["reaction_hooray"] = count
            elif content == "confused":
                comment_reactions["reaction_confused"] = count
            elif content == "heart":
                comment_reactions["reaction_heart"] = count
            elif content == "rocket":
                comment_reactions["reaction_rocket"] = count
            elif content == "eyes":
                comment_reactions["reaction_eyes"] = count

        comment_row.update(comment_reactions)

        comment_row["total_reactions"] = sum(comment_reactions.values())

        rows.append(comment_row)

    return rows


def process_all_repositories(data_dir):
    all_rows = []

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
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        issue_data = json.loads(line)
                        rows = flatten_issue_data(issue_data, repo_name)
                        all_rows.extend(rows)
                        issue_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {repo_name}: {e}")
                        continue

                print(f"  Processed {issue_count} issues from {repo_name}")

        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
            continue

    if not all_rows:
        print("No data processed from any repository")
        return pd.DataFrame()

    print(f"\nTotal rows created: {len(all_rows)}")

    df = pd.DataFrame(all_rows)

    datetime_columns = [
        "issue_created_at",
        "issue_updated_at",
        "issue_closed_at",
        "comment_created_at",
        "comment_updated_at",
    ]

    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def create_parquet_dataset():
    """Main function to create the Parquet dataset."""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    parquet_file = output_dir / "quantum_issues_dataset.parquet"

    print("Starting Parquet dataset creation...")
    print(f"Data directory: {data_dir}")
    print(f"Output file: {parquet_file}")

    df = process_all_repositories(data_dir)

    if df.empty:
        print("Error: No data was processed. Please check your data directory.")
        return

    print(f"\nDataset Statistics:")
    print(f"Total rows: {len(df):,}")
    print(f"Total repositories: {df['repo'].nunique()}")

    issue_rows = df[df["type"] == "issue"]
    comment_rows = df[df["type"] == "comment"]

    print(f"Total unique issues: {len(issue_rows):,}")
    print(f"Total comments: {len(comment_rows):,}")

    print(f"\nRepository breakdown:")
    repo_stats = (
        df.groupby("repo")
        .agg(
            {
                "issue_number": lambda x: x[
                    df.loc[x.index, "type"] == "issue"
                ].nunique(),
                "type": lambda x: (x == "comment").sum(),
            }
        )
        .rename(columns={"issue_number": "issues", "type": "comments"})
    )

    for repo, stats in repo_stats.iterrows():
        print(f"  {repo}: {stats['issues']} issues, {stats['comments']} comments")

    print(f"\nSaving to Parquet format...")

    try:
        table = pa.Table.from_pandas(df)

        pq.write_table(
            table,
            parquet_file,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )

        file_size = parquet_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        print(f"Successfully created Parquet dataset!")
        print(f"File: {parquet_file}")
        print(f"Size: {file_size_mb:.2f} MB")

        print(f"\nVerifying Parquet file...")
        test_table = pq.read_table(parquet_file)
        test_df = test_table.to_pandas()
        print(f"Verification successful: {len(test_df):,} rows read")

    except Exception as e:
        print(f"Error saving Parquet file: {e}")
        raise


if __name__ == "__main__":
    create_parquet_dataset()
