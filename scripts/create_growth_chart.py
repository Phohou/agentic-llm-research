import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

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

def loadDf():
    """Loads the dataset from the parquet file."""
    dataPath = Path("output/issues_filtered_quarter.parquet")

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
    # Define a consistent color mapping for known repositories
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

def createMonthlyGrowthPlot(df, date):
    """Creates and saves a semi-annual growth chart."""
    
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    repos = df['repo'].unique()
    
    # Get consistent color mapping for repositories
    repo_colors = get_repo_color_mapping(repos)
    
    for i, repo in enumerate(repos):
        # Filters the data to only issues from the current repository
        repoDf = df[df['repo'] == repo].copy()
        
        # Sort by date
        repoDf = repoDf.sort_values(date)

        # Convert the date column to timezone-naive datetime if it's timezone-aware
        if repoDf[date].dt.tz is not None:
            repoDf[date] = repoDf[date].dt.tz_localize(None)
        
        # Skip if no data remains after filtering
        if len(repoDf) == 0:
            continue

        # Create cumulative count starting from 0
        repoDf['cumulative_count'] = range(1, len(repoDf) + 1)
        
        # Group by quarter and get the last cumulative count for each quarter
        repoDf['year_quarter'] = repoDf[date].dt.to_period('Q')
        quarterly_data = repoDf.groupby('year_quarter').agg({
            date: 'last',  # Use the last date in the quarter
            'cumulative_count': 'last'  # Use the last cumulative count in the quarter
        }).reset_index(drop=True)
        
        # Add a starting point at (first_date, 0) to eliminate the gap
        first_date = quarterly_data[date].iloc[0]
        start_point = pd.DataFrame({
            date: [first_date],
            'cumulative_count': [0]
        })
        
        # Combine start point with actual data
        plot_data = pd.concat([start_point, quarterly_data[[date, 'cumulative_count']]], ignore_index=True)

        ax.plot(plot_data[date], plot_data['cumulative_count'], 
               label=repo, color=repo_colors[repo], linewidth=PLOT_LINE_WIDTH)
    
    # Customize the plot
    ax.set_ylabel('Number of Closed Issues', fontsize=FONT_SIZES['axis_label'])
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Set x-axis limit to end at July 2025
    ax.set_xlim(right=pd.to_datetime('2025-07-31'))
    
    # Format x-axis dates - show every 6 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Every 6 months
    plt.xticks(rotation=45)
    
    # Add legend with cleaned repo names
    cleaned_labels = []
    for repo in repos:
        cleaned_name = clean_repo_name(repo)
        cleaned_labels.append(cleaned_name)
    
    # Update legend labels - moved inside the plot
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, cleaned_labels, loc='upper left', framealpha=0.9,
              fontsize=FONT_SIZES['legend'], edgecolor='#CCCCCC')
    
    # Apply grid styling
    apply_grid_style(ax)
    
    return fig

def total_issues_repo(df, repo_col):
    """Create a bar chart showing total issues by repository with cleaned names"""
    
    setup_plotting_style()

    if repo_col:
        # Get repository counts
        repo_counts = df[repo_col].value_counts()
        
        # Create cleaned names for the repositories
        cleaned_labels = []
        for repo in repo_counts.index:
            cleaned_name = clean_repo_name(repo)
            cleaned_labels.append(cleaned_name)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart with cleaned names using color palette
        bars = ax.barh(cleaned_labels, repo_counts.values, 
                      color=PAIRED_COLORS[:len(repo_counts)])
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, repo_counts.values)):
            # Position text at the end of each bar
            if value >= 1000:
                label = f'{value/1000:.1f}K'
            else:
                label = f'{value}'
            
            ax.text(value + max(repo_counts.values) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   label, ha='left', va='center', fontsize=FONT_SIZES['annotation'], fontweight='bold')
        
        ax.set_xlabel('Number of Issues', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel('Repository', fontsize=FONT_SIZES['axis_label'])
        
        # Apply grid styling
        apply_grid_style(ax)
        
        # Format x-axis with 1K, 10K notation if needed
        def format_thousands(x, pos):
            if x >= 1000:
                return f'{x/1000:.1f}K'
            else:
                return f'{x:.0f}'
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
        
        plt.tight_layout()
        
    return fig

def plot_average_time_to_close_over_time(df, date_created, date_closed):
    """Plot the average time to close issues over time."""
    
    setup_plotting_style()

    # Create a copy of the dataframe to work with
    plot_df = df.copy()
    
    # Convert date columns to datetime if they're not already
    plot_df[date_created] = pd.to_datetime(plot_df[date_created])
    plot_df[date_closed] = pd.to_datetime(plot_df[date_closed])
    
    # Calculate time to close in days
    plot_df['time_to_close_days'] = (plot_df[date_closed] - plot_df[date_created]).dt.total_seconds() / (24 * 60 * 60)
    
    # Filter out negative values and extreme outliers (issues that took more than 2 years)
    plot_df = plot_df[(plot_df['time_to_close_days'] >= 0) & (plot_df['time_to_close_days'] <= 730)]
    
    # Create quarterly grouping for smoother trends
    plot_df['year_quarter'] = plot_df[date_created].dt.to_period('Q')
    
    # Calculate average time to close by quarter and repository
    quarterly_avg = plot_df.groupby(['repo', 'year_quarter'])['time_to_close_days'].agg(['mean', 'count']).reset_index()
    quarterly_avg['date'] = quarterly_avg['year_quarter'].dt.to_timestamp()
    
    # Filter out quarters with less than 5 issues for statistical significance
    quarterly_avg = quarterly_avg[quarterly_avg['count'] >= 5]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique repositories
    repos = quarterly_avg['repo'].unique()
    repo_colors = get_repo_color_mapping(repos)
    
    for i, repo in enumerate(repos):
        repo_data = quarterly_avg[quarterly_avg['repo'] == repo]
        
        # Clean repository name for legend
        cleaned_name = clean_repo_name(repo)
            
        ax.plot(repo_data['date'], repo_data['mean'], 
               label=cleaned_name.title(), color=repo_colors[repo], linewidth=PLOT_LINE_WIDTH)
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Average Time to Close (Days)', fontsize=FONT_SIZES['axis_label'])
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Cut off at July 2025 to match your other chart
    ax.set_xlim(right=pd.to_datetime('2025-07-31'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add legend with styling
    setup_legend(ax, loc='upper left')
    
    # Apply grid styling
    apply_grid_style(ax)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_overall_average_time_to_close(df, date_created, date_closed):
    """Plot overall average time to close across all repositories."""
    
    setup_plotting_style()

    # Create a copy of the dataframe to work with
    plot_df = df.copy()
    
    # Convert date columns to datetime
    plot_df[date_created] = pd.to_datetime(plot_df[date_created])
    plot_df[date_closed] = pd.to_datetime(plot_df[date_closed])
    
    # Calculate time to close in days
    plot_df['time_to_close_days'] = (plot_df[date_closed] - plot_df[date_created]).dt.total_seconds() / (24 * 60 * 60)
    
    # Filter out negative values and extreme outliers
    plot_df = plot_df[(plot_df['time_to_close_days'] >= 0) & (plot_df['time_to_close_days'] <= 730)]
    
    # Create quarterly grouping
    plot_df['year_quarter'] = plot_df[date_created].dt.to_period('Q')
    
    # Calculate overall average by quarter
    quarterly_avg = plot_df.groupby('year_quarter')['time_to_close_days'].agg(['mean', 'median', 'count']).reset_index()
    quarterly_avg['date'] = quarterly_avg['year_quarter'].dt.to_timestamp()
    
    # Filter out quarters with less than 10 issues
    quarterly_avg = quarterly_avg[quarterly_avg['count'] >= 10]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both mean and median with styling
    ax.plot(quarterly_avg['date'], quarterly_avg['mean'], 
           label='Average (Mean)', color=MAIN_COLORS[0], linewidth=PLOT_LINE_WIDTH * 1.5)
    ax.plot(quarterly_avg['date'], quarterly_avg['median'], 
           label='Median', color=MAIN_COLORS[1], linewidth=PLOT_LINE_WIDTH)
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Time to Close (Days)', fontsize=FONT_SIZES['axis_label'])
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Cut off at July 2025
    ax.set_xlim(right=pd.to_datetime('2025-07-31'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add legend with styling
    setup_legend(ax, loc='upper left')
    
    # Apply grid styling
    apply_grid_style(ax)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def percentage_labeled_issues(df, repo_col):
    """Create a bar chart showing percentage of labeled issues by repository."""
    
    setup_plotting_style()

    if repo_col:
        # Calculate percentage of labeled issues for each repository
        repo_stats = []
        
        for repo in df[repo_col].unique():
            repo_data = df[df[repo_col] == repo]
            total_issues = len(repo_data)
            labeled_issues = len(repo_data[repo_data['labels_count'] >= 1])
            percentage = (labeled_issues / total_issues) * 100 if total_issues > 0 else 0
            
            repo_stats.append({
                'repo': repo,
                'total_issues': total_issues,
                'labeled_issues': labeled_issues,
                'percentage': percentage
            })
        
        # Convert to DataFrame and sort by percentage
        stats_df = pd.DataFrame(repo_stats)
        stats_df = stats_df.sort_values('percentage', ascending=True)
        
        # Create cleaned names for the repositories
        cleaned_labels = []
        for repo in stats_df['repo']:
            cleaned_name = clean_repo_name(repo)
            cleaned_labels.append(cleaned_name)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart with cleaned names using color palette
        bars = ax.barh(cleaned_labels, stats_df['percentage'].values, 
                      color=PAIRED_COLORS[:len(stats_df)])
        
        # Add percentage labels on bars
        for i, (bar, percentage, labeled, total) in enumerate(zip(bars, stats_df['percentage'].values, 
                                                                 stats_df['labeled_issues'].values, 
                                                                 stats_df['total_issues'].values)):
            # Position text at the end of each bar
            label = f'{percentage:.1f}% ({labeled}/{total})'
            
            ax.text(percentage + 2, bar.get_y() + bar.get_height()/2, 
                   label, ha='left', va='center', fontsize=FONT_SIZES['annotation'], fontweight='bold')
        
        # Customize the plot
        ax.set_xlabel('Percentage of Issues with Labels (%)', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel('Repository', fontsize=FONT_SIZES['axis_label'])
        
        # Set x-axis limits (0-100% plus some space for labels)
        ax.set_xlim(0, max(stats_df['percentage'].values) + 15)
        
        # Apply grid styling
        apply_grid_style(ax)
        
        # Add percentage formatting to x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        plt.tight_layout()

    return fig

def createCumulativeGrowthPlot(df, date):
    """Creates and saves a cumulative growth chart."""
    
    setup_plotting_style()

    # Copy the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Convert the date column to datetime and remove the timezone
    plot_df[date] = pd.to_datetime(plot_df[date])
    if plot_df[date].dt.tz is not None:
        plot_df[date] = plot_df[date].dt.tz_localize(None)

    # Sort the dataframe by date
    plot_df = plot_df.sort_values(date)

    cutoff = pd.to_datetime('2025-07-31')
    plot_df = plot_df[plot_df[date] <= cutoff]
    plot_df['year_quarter'] = plot_df[date].dt.to_period('Q')

    quarterly_counts = plot_df.groupby('year_quarter').size().reset_index(name='issue_count')
    quarterly_counts['cumulative_count'] = quarterly_counts['issue_count'].cumsum()
    quarterly_counts['date'] = quarterly_counts['year_quarter'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Add a starting point at (first_date, 0)
    first_date = quarterly_counts['date'].iloc[0]
    start_point = pd.DataFrame({
        'date': [first_date],
        'cumulative_count': [0]
    })
    
    # Combine start point with actual data
    plot_data = pd.concat([start_point, quarterly_counts[['date', 'cumulative_count']]], ignore_index=True)
    
    # Plot the cumulative growth with styling
    ax.plot(plot_data['date'], plot_data['cumulative_count'], 
           color=MAIN_COLORS[0], linewidth=PLOT_LINE_WIDTH * 2)
    
    # Customize the plot
    ax.set_xlabel('Quarter', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Cumulative Number of Closed Issues', fontsize=FONT_SIZES['axis_label'])
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Set x-axis limit to end at July 2025
    ax.set_xlim(right=pd.to_datetime('2025-07-31'))
    
    # Format x-axis dates - show quarters
    def quarter_formatter(x, pos):
        date = mdates.num2date(x)
        quarter = (date.month - 1) // 3 + 1
        return f'{date.year} Q{quarter}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.xticks(rotation=45)
    
    # Format y-axis with 1K, 10K notation
    def format_thousands(x, pos):
        if x >= 1000:
            return f'{x/1000:.0f}K'
        else:
            return f'{x:.0f}'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    # Apply grid styling
    apply_grid_style(ax)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def main():
    try:
        # Load the dataset
        df = loadDf()
        if df is None:
            return

        # Create output directory if it doesn't exist
        outputDir = Path("output")
        outputDir.mkdir(parents=True, exist_ok=True)

        # Plot monthly growth chart
        print("\nCreating growth chart...")
        fig1 = createMonthlyGrowthPlot(df, 'issue_created_at')
        fig1.savefig(outputDir / "monthly_growth_chart2.png")
        print(f"Growth chart saved to {outputDir / 'monthly_growth_chart2.png'}")
        plt.close(fig1)

        # Plot total issues by repository
        print("\nCreating total issues by repository chart...")
        fig2 = total_issues_repo(df, 'repo')
        fig2.savefig(outputDir / "total_issues_by_repository2.png")
        print(f"Total issues by repository chart saved to {outputDir / 'total_issues_by_repository2.png'}")
        plt.close(fig2)

        # Plot average time to close by repository
        fig2 = plot_average_time_to_close_over_time(df, 'issue_created_at', 'issue_closed_at')
        fig2.savefig(outputDir / "average_time_to_close_by_repo2.png", dpi=300, bbox_inches='tight')
        print(f"Average time to close by repo chart saved to {outputDir / 'average_time_to_close_by_repo2.png'}")
        plt.close(fig2)
                
        # Plot overall average
        fig3 = plot_overall_average_time_to_close(df, 'issue_created_at', 'issue_closed_at')
        fig3.savefig(outputDir / "overall_average_time_to_close2.png", dpi=300, bbox_inches='tight')
        print(f"Overall average time to close chart saved to {outputDir / 'overall_average_time_to_close2.png'}")
        plt.close(fig3)

        # Plot percentage of labeled issues
        print("\nCreating percentage labeled issues chart...")
        fig4 = percentage_labeled_issues(df, 'repo')
        fig4.savefig(outputDir / "percentage_labeled_issues2.png", dpi=300, bbox_inches='tight')
        print(f"Percentage labeled issues chart saved to {outputDir / 'percentage_labeled_issues2.png'}")
        plt.close(fig4)

        print("\nCreating cumulative quarterly growth chart...")
        fig_quarterly = createCumulativeGrowthPlot(df, 'issue_created_at')
        fig_quarterly.savefig(outputDir / "cumulative_quarterly_growth2.png", dpi=300, bbox_inches='tight')
        print(f"Cumulative quarterly growth chart saved to {outputDir / 'cumulative_quarterly_growth2.png'}")
        plt.close(fig_quarterly)
                
        # Print some statistics
        df['time_to_close_days'] = (pd.to_datetime(df['issue_closed_at']) - pd.to_datetime(df['issue_created_at'])).dt.total_seconds() / (24 * 60 * 60)
        print(f"\nTime to Close Statistics:")
        print(f"Average time to close: {df['time_to_close_days'].mean():.1f} days")
        print(f"Median time to close: {df['time_to_close_days'].median():.1f} days")
        print(f"Min time to close: {df['time_to_close_days'].min():.1f} days")
        print(f"Max time to close: {df['time_to_close_days'].max():.1f} days")

        # Display summary statistics

        print(f"Unique repositories: {df['repo'].unique()}")
        print(f"Total issues: {len(df)}")
        print(f"Issues per repository:\n {df['repo'].value_counts()}")

    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# TODO:
# Get a quarterly dataset or create a function that plots it by quarter in order to smooth out the curves
# Fix the 0 start issue from the y axis plot
# Properly label the axes and add a title
# Can use 1k.... to save space on labeling if needed if there is a lot of data to plot at hand
# Style the plot properly
# Change legend names to be proper repo names instead of full github repo names (can add the the legend to blank spots of the graph)
# Consult the files that use the ieee import
# Use pointers on points of interest on the graph if needed, better to do it on the graph (annoying to do in code so can do it in post if needed)

# Pull request
# Look at how long issues are taken to be fixed
# Look at the average time an issue takes to be fixed
# Want to see which repos are contributing more in the first half or other half *
# Tie issue count with how long each repository is taking to close their issues (finding correlation)
# Look at the contributors (see if the more popular repos have more contributors or not)
# Find the percentage of issues that have labels can split the data for a study (if the labels are similar gentrify it)
# Can look at which type of isses have more replies (bug, feature request, etc)
# Look at what the feature requests are (use use llm to summarize the feature requests, fine tune and structure the output, watch for the words it tries to use)
# Clustering the data (can use the embeddings from openai to cluster the data)

# Exact unique values for labels get a set for all the labels gentrify them and merge the similar ones
# Get the data to be more readable
# Look at the size of the pr fix, get the diff file that has line deleted, inserted, replaced, etc.
# See if there is a correlation over the complexity of the bugs and the time it takes over time

# Remove the title
# Thicker lines
# Change figure size