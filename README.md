# Multi-Agent Systems Framework Analysis

## Overview

This repository supports a large-scale empirical analysis that examines:

- Development patterns across major MAS frameworks
- Commit activity, code churn, and maintenance behavior
- Issue reporting trends and resolution efficiency
- Dominant technical challenges in agentic AI systems

Our dataset includes **42,000+ unique commits** and **~4,700 resolved issues** collected from widely used open-source MAS frameworks.

The study analyzes the following representative multi-agent frameworks:

- AutoGen  
- CrewAI  
- Haystack  
- LangChain  
- Letta  
- LlamaIndex  
- Semantic Kernel  
- SuperAGI  

---

## Repository Structure

```
├── data/          # Raw and processed datasets
├── scripts/       # Data collection and analysis scripts
├── figures/       # Plots used in the paper and presentation slides
├── output/        # Tables, summaries, and intermediate results
└── utils/         # Shared helper functions and utilities
```

---

## Setup

### Prerequisites

- **Python 3.11+** (for analysis scripts)
- **Deno** (for data collection scripts)

### Installation

#### 1. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Python dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly pyarrow
```

## Scripts

### Data Collection (TypeScript/Deno)

**Issue data collection:**
```bash
deno run --allow-net --allow-read --allow-write --allow-env scripts/get_issues.ts
deno run --allow-net --allow-read --allow-write --allow-env scripts/get_issues_comments.ts
deno run --allow-net --allow-read --allow-write --allow-env scripts/get_closing_pr.ts
```

**Generate reports:**
```bash
deno run --allow-read --allow-write scripts/generate_csv_report.ts
deno run --allow-read --allow-write scripts/generate_text_files.ts
```

**Extract samples:**
```bash
deno run --allow-read --allow-write scripts/extract_sample.ts
deno run --allow-read --allow-write scripts/extract_sample_regex.ts
```

### Data Processing (Python)

**Create Parquet datasets:**
```bash
python scripts/create_parquet_dataset.py
python scripts/create_parquet_dataset_nested.py
python scripts/create_parquet_dataset_separated.py
```

**Dataset statistics:**
```bash
python scripts/get_dataset_stats.py
python scripts/create_repository_summary.py
```

**Extract prominent items:**
```bash
python scripts/get_prominent_issues.py
python scripts/get_prominent_labels.py
```

### Visualization (Python)

**Commit analysis:**
```bash
python scripts/create_commits_analysis.py
python scripts/create_commits_heat_map.py
python scripts/create_commits_sparklines.py
python scripts/create_commits_trend_no_agg.py
python scripts/create_commits_trend_percentage.py
python scripts/create_commit_cv_chart.py
python scripts/create_commit_scatter_plots.py
python scripts/create_commit_stat_trend.py
```

**Code churn analysis:**
```bash
python scripts/create_add_delete_ratio_chart.py
python scripts/create_churn_boxplots.py
python scripts/create_code_churn_comparison.py
```

**Growth and contribution:**
```bash
python scripts/create_growth_chart.py
python scripts/create_growth_chart_no_agg.py
python scripts/create_contribution_vs_consistency.py
python scripts/create_stacked_area_chart.py
```

**Issue analysis:**
```bash
python scripts/create_issue_heat_map.py
python scripts/create_issue_label_charts.py
python scripts/create_issue_resolution_chart.py
python scripts/create_labeled_chart.py
```

**Pull request analysis:**
```bash
python scripts/create_pr_chart.py
```
