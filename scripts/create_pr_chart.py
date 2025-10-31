import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import pyarrow.parquet as pq

from plot_utils import (
    GREY_COLORS_DARK,
    setup_plotting_style,
    MAIN_COLORS,
    PAIRED_COLORS,
    FIG_SIZE_SINGLE_COL,
    PLOT_LINE_WIDTH,
    setup_axis_ticks,
    setup_legend,
    save_plot,
    create_pie_chart,
    apply_grid_style,
    FONT_SIZES,
)

def clean_repo_name(repo):
    """Cleans the repository name for better readability."""
    cleaned_name = repo.split('/')[-1]
    if '-' in cleaned_name or '_' in cleaned_name:
        cleaned_name = cleaned_name.replace('-', ' ')
        cleaned_name = cleaned_name.replace('_', ' ')
        cleaned_name = cleaned_name.title()
    return cleaned_name[0].upper() + cleaned_name[1:]

def get_repo_color_mapping(repos):
    """Create a consistent color mapping for repositories."""

    repo_color_map = {
        'langchain-ai/langchain': MAIN_COLORS[0],
        'run-llama/llama_index': MAIN_COLORS[1],
        'microsoft/autogen': MAIN_COLORS[2],
        'deepset-ai/haystack': MAIN_COLORS[3],
        'crewAIInc/crewAI': MAIN_COLORS[4],
        'microsoft/semantic-kernel': MAIN_COLORS[5],
        'TransformerOptimus/SuperAGI': MAIN_COLORS[6],
        'letta-ai/letta': MAIN_COLORS[7],
        'FoundationAgents/MetaGPT': MAIN_COLORS[8],
    }
    
    # For any repos not in the predefined map, assign colors from the remaining palette
    colors = {}
    used_colors = set()
    
    for repo in repos:
        if repo in repo_color_map:
            colors[repo] = repo_color_map[repo]
            used_colors.add(repo_color_map[repo])
        else:
            # Find an unused color
            for color in MAIN_COLORS:
                if color not in used_colors:
                    colors[repo] = color
                    used_colors.add(color)
                    break
    
    return colors

def loadDataFrame():
    """Loads the dataset from the parquet file."""
    dataPath = Path("output/latestcommits/combined_commits_deduped.parquet")

    if not dataPath.exists():
        print(f"File {dataPath} not found.")
        return None
    
    print(f"Loading data from {dataPath}")
    df = pd.read_parquet(dataPath)
    
    # Checking the structure of the DataFrame
    print(f"Loaded {len(df)} issues")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")

    return df

def createPRLinePlot(df):
    """Creates PR Plot showing average lines changed per month for all repositories"""

    setup_plotting_style()

    # Convert committer_date to datetime if not already
    df['committer_date'] = pd.to_datetime(df['committer_date'])
    
    # Calculate total lines changed (insertions + deletions)
    df['lines_changed'] = df['insertions'] + df['deletions']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique repositories and color mapping
    repos = df["repo_name"].unique()
    repo_colors = get_repo_color_mapping(repos)

    for repo in repos:
        repo_df = df[df["repo_name"] == repo].copy()
        
        # Set committer_date as index for resampling
        repo_df = repo_df.set_index('committer_date')
        
        # Group by month and calculate average lines changed per PR
        monthly_avg = repo_df['lines_changed'].resample('M').mean()
        
        # Remove NaN values (months with no PRs)
        monthly_avg = monthly_avg.dropna()
        
        # Get cleaned name for legend
        cleaned_name = clean_repo_name(repo)

        # Plot this repository's data
        ax.plot(
            monthly_avg.index,
            monthly_avg.values,
            color=repo_colors[repo],
            linewidth=PLOT_LINE_WIDTH,
            label=cleaned_name
        )

    # Customize the plot
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Average Lines Changed per PR', fontsize=FONT_SIZES['axis_label'])
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add legend with cleaned repo names
    ax.legend(loc='upper left', framealpha=0.9,
              fontsize=FONT_SIZES['legend'], edgecolor='#CCCCCC')
    
    # Apply grid styling
    apply_grid_style(ax)
    
    # Adjust layout
    plt.tight_layout()

    return fig

def main():
    df = loadDataFrame()
    if df is None:
        return
    
    outputDir = Path("output")
    outputDir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating PR average lines changed chart...")
    fig = createPRLinePlot(df)
    fig.savefig(outputDir / "pr_avg_lines_changed_all_repos.png", dpi=300, bbox_inches='tight')
    print(f"PR chart saved to {outputDir / 'pr_avg_lines_changed_all_repos.png'}")
    plt.close(fig)

if __name__ == "__main__":
    main()