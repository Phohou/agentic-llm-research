import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import pyarrow.parquet as pq

def loadDataFrame():
    """Loads the dataset from the parquet file."""
    dataPath = Path("output/issues_with_categorized_labels_prs.parquet")

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



def get_label_categories():
    """Define the simplified label categorization."""
    return {
        'Bug': [
            'type:bug',
            'bug'
        ],
        
        'Feature': [
            'type:feature',
            'feature-request', 'feature request',
            'experimental', 'feature_graduation', 'feature branch',
            'optimization', 'multimodal', 'structured output'
        ],
        
        'Documentation': [
            'type:documentation', 'documentation', 'docs', 'docs and tests'
        ],
        
        'Infrastructure': [
            'type:refactor', 'topic:core', 'topic:pipeline', 'topic:CI',  
            'topic:dependencies', 'topic:build/distribution', 'topic:deployment', 
            'topic:docker', 'github_actions', 'topic:windows',
            'topic:security', 'topic:tests', 'topic:streaming',
            'core plugin', 'dependencies', 'processes', 'Build', 'aot',
            'cloud-events',
            'proj-core', 'proj-extensions', 'proj-studio', 'proj-autogenbench', 
            'proj-agentchat', 'proj-magentic-one', 'code-execution', 'autobuilder',
            'logging', 'azure',
            'ai connector',
            'kernel', 'kernel.core',
        ],
        
        'Data Processing': [
            'topic:preprocessing', 'topic:indexing', 'topic:metadata', 
            'topic:file_converter', 'topic:reader', 'pdf', 'topic:images', 
            'topic:audio', 'topic:crawler', 'topic:save/load',
            'memory', 'memory connector', 'text_search', 'filters',
            'rag', 'topic:vector stores', 'msft.ext.vectordata',
            'topic:DPR', 'topic:dc_document_store', 'topic:document_store', 'topic:elasticsearch',
            'topic:faiss', 'topic:opensearch', 'topic:pinecone', 'topic:retriever',
            'topic:sql', 'topic:weaviate'
        ],
        
        'Agent Related Issues & Implementations': [
            'topic:agent', 'topic:promptnode', 'topic:LLM', 'topic:modeling', 
            'topic:predictions', 'topic:train', 'topic:labelling', 
            'topic:knowledge_graph', 'topic:tableQA',
            'topic:eval', 'agents', 'multi-agent', 'chat history',
            'planner', 'function_calling', 'mcp', 'modelcontextprotocol',
            'prompty', 'models', 'tool-usage', 'group chat/teams',
            'msft.ext.ai', # semantic-kernel label for issues relating to microsoft.ext.ai
            'openai_sdk_v2', 'openapi',
            'topic:accuracy'
        ],
        
        'Project Workflow': [
            'epic', 'epic:idle', 'epic:abandoned', 'epic:in-progress', 
            'wontfix', 'duplicate', 'stale', 'proposal',
            'ignore-for-release-notes', 'information-needed',
            'priority', 'roadmap', 'topic:workflows',
            'PR: in progress', 'PR: ready for review', 'PR: ready to merge',
            'SK-H2-Planning', 'blocked', 'blocked external',
            'release blocker', 'sk team issue', 'needs-design',
            'awaiting-op-response',
            'size-small', 'size-medium', 'size-large',
            'api-break-change',
            'great-writeup', # Used to highlight well-documented issues in langchain
            'auto-closed',
            'beta', 'needs_port_to_dotnet', 'needs_port_to_python',
            'breaking change'
        ],

        'Triage': [
            'P0', 'P1', 'P2', 'P3',
            'triage', 'needs-triage',
        ],

        'User Experience': [
            'topic:DX', 'topic:cli', 'topic:rest_api',
            'api', 'samples', 'vscode', 'ChatUI', 'telemetry', 'topic:telemetry',
            'topic:installation'
        ],

        'Community Engagement': [
            'help wanted', 'no-issue-activity', 'investigate', 'follow up', 'question',
            'needs help', 'needs more info', 'needs discussion', 'Contributions wanted!',
            'good first issue'
        ],

        'Language or Framework Specific': [
            '.NET', 'java', 'python', 'language:python'
        ],

        'Excluded': [
            'enhancement', # Label is ambiguous, appears in multiple categories
            'type:enhancement',
        ]

    }

def categorize_label(label, label_categories):
    """Categorize a single label into one of the simplified categories."""
    for category, labels in label_categories.items():
        if label in labels:
            return category
    return 'Other'  # For labels not in our categorization

def create_repo_folder(repo_name, base_output_dir):
    """Create a folder for the repository and return the path."""
    # Clean up repository name for folder
    clean_repo_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
    
    # Remove any characters that might be problematic for folder names
    clean_repo_name = "".join(c for c in clean_repo_name if c.isalnum() or c in ('-', '_', '.'))
    
    # Create the repository folder
    repo_folder = base_output_dir / clean_repo_name
    repo_folder.mkdir(parents=True, exist_ok=True)
    
    return repo_folder, clean_repo_name

def plot_categorized_labels(df, repo_name, label_categories):
    """Plot issues by categorized labels for a specific repository."""
    
    # Filter to specific repository and labeled issues
    repo_data = df[(df['repo'] == repo_name) & (df['labels_count'] > 0)]
    
    if len(repo_data) == 0:
        print(f"No labeled issues found for {repo_name}")
        return None
    
    # Explode labels and categorize them
    exploded_labels = repo_data['labels'].explode()
    categorized_labels = exploded_labels.apply(lambda x: categorize_label(x, label_categories))
    
    # Count categories
    category_counts = categorized_labels.value_counts()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart for better readability
    bars = plt.barh(category_counts.index, category_counts.values)
    
    # Color the bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for i, (category, count) in enumerate(category_counts.items()):
        plt.text(count + max(category_counts.values) * 0.01, i, 
                str(count), va='center', fontweight='bold')
    
    # Clean up repository name for title
    clean_repo_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
    
    plt.xlabel('Number of Issues', fontsize=12)
    plt.ylabel('Label Category', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_detailed_breakdown(df, repo_name, label_categories):
    """Plot detailed breakdown showing original labels within each category."""
    
    # Filter to specific repository and labeled issues
    repo_data = df[(df['repo'] == repo_name) & (df['labels_count'] > 0)]
    
    if len(repo_data) == 0:
        return None
    
    # Get all labels and their counts
    all_labels = repo_data['labels'].explode()
    label_counts = all_labels.value_counts()
    
    # Create a mapping of labels to categories
    label_to_category = {}
    for category, labels in label_categories.items():
        for label in labels:
            if label in label_counts.index:
                label_to_category[label] = category
    
    # Group by category
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    categories = list(label_categories.keys())
    
    for i, category in enumerate(categories):
        if i >= len(axes):
            break
            
        # Get labels for this category
        category_labels = [label for label, cat in label_to_category.items() if cat == category]
        if not category_labels:
            axes[i].set_visible(False)
            continue
            
        category_counts = label_counts[category_labels].sort_values(ascending=True)
        
        # Create subplot
        bars = axes[i].barh(range(len(category_counts)), category_counts.values)
        axes[i].set_yticks(range(len(category_counts)))
        axes[i].set_yticklabels(category_counts.index, fontsize=8)
        axes[i].set_title(f'{category.title()} ({category_counts.sum()} total)', fontweight='bold')
        axes[i].set_xlabel('Count')
        
        # Add value labels
        for j, count in enumerate(category_counts.values):
            axes[i].text(count + max(category_counts.values) * 0.02, j, 
                        str(count), va='center', fontsize=8)
        
        axes[i].grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for i in range(len(categories), len(axes)):
        axes[i].set_visible(False)
    
    clean_repo_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
    fig.suptitle(f'Detailed Label Breakdown for {clean_repo_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_by_labels(df, base_output_dir):
    """Plot issues by categorized labels for each repository."""
    
    label_categories = get_label_categories()
    labeled_issues = df[df['labels_count'] > 0]
    
    for repo in labeled_issues['repo'].unique():
        print(f"\nProcessing Repository: {repo}")
        
        # Create repository folder
        repo_folder, clean_repo_name = create_repo_folder(repo, base_output_dir)
        print(f"Created folder: {repo_folder}")
        
        # Print original unique labels
        repo_labels = labeled_issues[labeled_issues['repo'] == repo]['labels'].explode().unique()
        print(f"Total unique labels: {len(repo_labels)}")
        
        # Show categorization summary
        categorized_counts = {}
        for label in repo_labels:
            category = categorize_label(label, label_categories)
            categorized_counts[category] = categorized_counts.get(category, 0) + 1
        
        print("Labels per category:")
        for category, count in sorted(categorized_counts.items()):
            print(f"  {category}: {count} unique labels")
        
        # Create categorized plot
        fig1 = plot_categorized_labels(df, repo, label_categories)
        if fig1:
            categorized_path = repo_folder / "categorized_labels.png"
            fig1.savefig(categorized_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"  Saved categorized chart: {categorized_path}")
        
        # Create detailed breakdown plot
        fig2 = plot_detailed_breakdown(df, repo, label_categories)
        if fig2:
            detailed_path = repo_folder / "detailed_labels_breakdown.png"
            fig2.savefig(detailed_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved detailed chart: {detailed_path}")
        
        # Create a summary text file with statistics
        summary_path = repo_folder / "label_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Label Analysis Summary for {clean_repo_name}\n")
            f.write(f"Total unique labels: {len(repo_labels)}\n")
            f.write(f"Total labeled issues: {len(labeled_issues[labeled_issues['repo'] == repo])}\n\n")
            
            f.write("Labels per category:\n")
            for category, count in sorted(categorized_counts.items()):
                f.write(f"  {category}: {count} unique labels\n")
            
            f.write(f"\nAll unique labels:\n")
            for label in sorted(repo_labels):
                category = categorize_label(label, label_categories)
                f.write(f"  {label} -> {category}\n")
        
        print(f"  Saved summary: {summary_path}")
        print(f"Completed processing for {clean_repo_name}")

def create_new_parquet(df, output_path):
    """Create a new parquet file with the labeled issues."""

    # Create a copy of the DataFrame
    dfLabeled = df.copy()

    dfLabeled['categorized_labels'] = dfLabeled['labels'].apply(
        lambda labels: [categorize_label(label, get_label_categories()) for label in labels]
    )

    dfLabeled.to_parquet(output_path, index=False)

    print(f"Saved new parquet file with labeled issues to {output_path}")

def main():
    try:
        # Load the dataset
        df = loadDataFrame()
        if df is None:
            return

        # Create main output directory
        base_output_dir = Path("output") / "repository_labels"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created main output directory: {base_output_dir}")

        # Plot issues by categorized labels
        print("\nAnalyzing issues by categorized labels...")
        plot_by_labels(df, base_output_dir)
        
        # Display summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print(f"Unique repositories: {len(df['repo'].unique())}")
        print(f"Repository names: {list(df['repo'].unique())}")
        print(f"Total issues: {len(df)}")
        print(f"Issues per repository:")
        for repo, count in df['repo'].value_counts().items():
            clean_name = repo.split('/')[-1] if '/' in repo else repo
            print(f"  {clean_name}: {count}")
        
        # Create new parquet file with categorized labels
        new_parquet_path = Path("output/issues_with_categorized_labels_prs.parquet")
        create_new_parquet(df, new_parquet_path)
        print(f"Created new parquet file with categorized labels: {new_parquet_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()