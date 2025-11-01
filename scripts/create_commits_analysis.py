import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.signal import savgol_filter

from plot_utils import (
    PLOT_LINE_WIDTH_THIN,
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
    MAIN_COLORS,
)

from constants import DATA_DIR, REPO_NAME_TO_DISPLAY

def load_data():
    dataPath = DATA_DIR / "combined_commits_deduped.parquet"
    df = pd.read_parquet(dataPath)
    return df

def create_commits_analysis_plot(df, date):

    setup_plotting_style()

    df = df.copy()

    if df[date].dt.tz is not None:
        df[date] = df[date].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=(12, 6))

    df = df.set_index(date)
    monthly_avg_insertions = df['insertions'].resample('ME').mean()
    monthly_avg_insertions = monthly_avg_insertions.dropna()
    monthly_avg_deletions = df['deletions'].resample('ME').mean()
    monthly_avg_deletions = monthly_avg_deletions.dropna()
    monthly_avg_files_changed = df['files_changed'].resample('ME').mean()
    monthly_avg_files_changed = monthly_avg_files_changed.dropna()

    span = 2   
    insertions_smoothed = monthly_avg_insertions.ewm(span=span, adjust=False).mean()
    deletions_smoothed = monthly_avg_deletions.ewm(span=span, adjust=False).mean()
    files_changed_smoothed = monthly_avg_files_changed.ewm(span=span, adjust=False).mean()

    ax.plot(
        insertions_smoothed.index,
        insertions_smoothed.values,
        color=MAIN_COLORS[0],
        linewidth=2,
        label='All Repositories - Insertions'
    )

    ax.plot(
        deletions_smoothed.index,
        deletions_smoothed.values,
        color=MAIN_COLORS[1],
        linewidth=2,
        label='All Repositories - Deletions'
    )

    ax.plot(
        files_changed_smoothed.index,
        files_changed_smoothed.values,
        color=MAIN_COLORS[2],
        linewidth=2,
        label='All Repositories - Files Changed'
    )

    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Average Lines Changed per Commit', fontsize=FONT_SIZES['axis_label'])
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)  # Changed from 0 to 1 since log scale can't start at 0

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper left', framealpha=0.9,
              fontsize=FONT_SIZES['legend'], edgecolor='#CCCCCC')
    
    apply_grid_style(ax)
    plt.tight_layout()

    return fig

def main():
    df = load_data()

    if df is None:
        print("Failed to load data")
        return
    
    outputDir = Path("output")
    figuresDir = Path("figures") 
    
    print("\nCreating commits analysis chart...")
    fig = create_commits_analysis_plot(df, 'author_date')
    fig.savefig(outputDir / "daily_avg_lines.png", dpi=300, bbox_inches='tight')
    fig.savefig(figuresDir / "commits_avg_lines_changed.pdf", bbox_inches='tight', format='pdf')
    print(f"Commits analysis chart saved to {outputDir / 'daily_avg_lines.png'}")
    plt.close(fig)

if __name__ == "__main__":
    main()