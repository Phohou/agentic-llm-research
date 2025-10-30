import pandas as pd
from constants import DATA_DIR

df = pd.read_parquet(DATA_DIR / "issues_with_categorized_labels_nocutoff.parquet")
labels = df["categorized_labels"].explode()
counts = labels.value_counts()
pct = (counts / len(df)) * 100
print(pd.DataFrame({"count": counts, "percentage": pct}))
