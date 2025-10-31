import pandas as pd
from constants import DATA_DIR

LABELS_TO_DISPLAY_FORMAT = {
    "Bug": "Bug",
    "Infrastructure": "Infrastructure",
    "Agent Related Issues & Implementations": "Agent Issues",
    "Data Processing": "Data Processing",
    "Documentation": "Documentation",
    "Feature": "Feature",
    "Community Engagement": "Community",
    "Language or Framework Specific": "Language/Framework",
    "Project Workflow": "Project Workflow",
    "Triage": "Triage",
    "Other": "Other",
    "Excluded": "Excluded",
    "User Experience": "User Experience",
}

df = pd.read_parquet(DATA_DIR / "issues_with_categorized_labels_nocutoff.parquet")
labels = df["categorized_labels"].explode()
counts = labels.value_counts()
pct = ((counts / len(df)) * 100).round(0).astype(int)

result_df = pd.DataFrame({"count": counts, "percentage": pct})
result_df.index = result_df.index.map(lambda x: LABELS_TO_DISPLAY_FORMAT.get(x, x))
print(result_df)
