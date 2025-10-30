# From the script get_prominent_labels.py,
# we got prominent labels in issues.
# Now, we want to get prominent issues that relate to those labels.

import pandas as pd

from constants import DATA_DIR


def main():
    labels_to_extract = ["Bug", "Language or Framework Specific"]

    df = pd.read_parquet(DATA_DIR / "issues_with_categorized_labels_nocutoff.parquet")

    filtered_issues_df = df[
        df["categorized_labels"].apply(
            lambda labels: any(label in labels_to_extract for label in labels)
        )
    ].copy()

    # Add a field prominent_label to indicate which prominent label(s) the issue has

    filtered_issues_df["prominent_labels"] = filtered_issues_df[
        "categorized_labels"
    ].apply(
        lambda labels: list(
            set(label for label in labels if label in labels_to_extract)
        )
    )

    filtered_issues_df.to_parquet(DATA_DIR / "prominent_issues.parquet", index=False)


if __name__ == "__main__":
    main()
