import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from plot_utils import (
    setup_plotting_style,
    FIG_SIZE_MEDIUM,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
)

from constants import FIGURES_DIR, OUTPUT_DIR, ROOT_DIR, DATA_DIR, REPO_NAME_TO_DISPLAY


def load_df():
    df = pd.read_parquet(DATA_DIR / "issues_filtered.parquet")
    return df


def calculate_resolution_days(df):
    df["resolution_time"] = df["issue_closed_at"] - df["issue_created_at"]
    df["resolution_days"] = df["resolution_time"].dt.total_seconds() / (24 * 3600)
    return df


def convert_to_naive_datetime(df):
    if df["issue_created_at"].dt.tz is not None:
        df["issue_created_at"] = df["issue_created_at"].dt.tz_convert(None)
    if df["issue_closed_at"].dt.tz is not None:
        df["issue_closed_at"] = df["issue_closed_at"].dt.tz_convert(None)
    return df


def prepare_resolution_data(df):
    closed_df = df[df["issue_closed"] == True].copy()
    closed_df = calculate_resolution_days(closed_df)
    closed_df = convert_to_naive_datetime(closed_df)
    return closed_df


def calculate_resolution_statistics(df):
    stats = (
        df.groupby("repo")["resolution_days"]
        .agg(
            [
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("count", "count"),
                ("min", "min"),
                ("max", "max"),
                ("25th_percentile", lambda x: x.quantile(0.25)),
                ("75th_percentile", lambda x: x.quantile(0.75)),
            ]
        )
        .round(2)
    )
    stats = stats.sort_values("median")
    return stats


def identify_and_save_zero_day_issues(df, output_dir):
    """Identify issues with 0 or very low resolution days and save them for investigation."""
    # Find issues with resolution_days that round to 0 (less than 0.5 days / 12 hours)
    zero_day_issues = df[df["resolution_days"] < 0.5].copy()

    if len(zero_day_issues) > 0:
        # Calculate hours for better understanding
        zero_day_issues["resolution_hours"] = (
            zero_day_issues["resolution_time"].dt.total_seconds() / 3600
        )
        zero_day_issues["resolution_minutes"] = (
            zero_day_issues["resolution_time"].dt.total_seconds() / 60
        )

        # Select relevant columns for investigation
        cols_to_save = [
            "repo",
            "issue_number",
            "issue_title",
            "issue_created_at",
            "issue_closed_at",
            "resolution_days",
            "resolution_hours",
            "resolution_minutes",
            "issue_url",
        ]
        # Only include columns that exist in the dataframe
        cols_to_save = [col for col in cols_to_save if col in zero_day_issues.columns]

        zero_day_issues_export = zero_day_issues[cols_to_save].copy()
        zero_day_issues_export = zero_day_issues_export.sort_values("resolution_days")

        # Save to CSV
        zero_day_issues_export.to_csv(
            output_dir / "zero_day_resolution_issues.csv", index=False
        )

        # Save summary to text file
        with open(output_dir / "zero_day_resolution_issues_summary.txt", "w") as f:
            f.write("Issues with Resolution Time < 0.5 Days (12 hours)\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total issues found: {len(zero_day_issues)}\n\n")
            f.write("Distribution by repo:\n")
            f.write(zero_day_issues["repo"].value_counts().to_string())
            f.write("\n\n")
            f.write("Statistics:\n")
            f.write(
                f"  Min resolution time: {zero_day_issues['resolution_days'].min():.6f} days\n"
            )
            f.write(
                f"  Max resolution time: {zero_day_issues['resolution_days'].max():.6f} days\n"
            )
            f.write(
                f"  Mean resolution time: {zero_day_issues['resolution_days'].mean():.6f} days\n"
            )
            f.write(
                f"  Median resolution time: {zero_day_issues['resolution_days'].median():.6f} days\n"
            )
            f.write("\n")
            f.write(
                f"  Min resolution hours: {zero_day_issues['resolution_hours'].min():.2f} hours\n"
            )
            f.write(
                f"  Max resolution hours: {zero_day_issues['resolution_hours'].max():.2f} hours\n"
            )
            f.write("\n\nSee 'zero_day_resolution_issues.csv' for detailed records.\n")

        print(f"Found {len(zero_day_issues)} issues with resolution time < 0.5 days")
        print(f"Details saved to: {output_dir / 'zero_day_resolution_issues.csv'}")
    else:
        print("No issues found with resolution time < 0.5 days")


def save_statistics_table(stats, output_dir):
    stats.to_csv(output_dir / "issue_resolution_statistics.csv")
    stats_display = stats.copy()
    stats_display.index = stats_display.index.map(
        lambda x: REPO_NAME_TO_DISPLAY.get(x, x)
    )
    with open(output_dir / "issue_resolution_statistics.txt", "w") as f:
        f.write("Issue Resolution Time Statistics (in days)\n")
        f.write("=" * 100 + "\n\n")
        f.write(stats_display.to_string())
        f.write("\n\n")
        f.write("Note: Resolution time = issue_closed_at - issue_created_at\n")
        f.write("Only closed issues are included in this analysis.\n")


def prepare_boxplot_data(df):
    # Sort repositories alphabetically by their display name (owner/repo -> display name)
    # This keeps ordering consistent with other charts that use REPO_NAME_TO_DISPLAY
    repos = sorted(df["repo"].unique(), key=lambda x: REPO_NAME_TO_DISPLAY.get(x, x))
    repo_display_names = [REPO_NAME_TO_DISPLAY.get(repo, repo) for repo in repos]
    data = [df[df["repo"] == repo]["resolution_days"].values for repo in repos]
    colors = get_repo_color_mapping(repos)
    box_colors = [colors[repo] for repo in repos]
    return repos, repo_display_names, data, box_colors


def plot_resolution_time_boxplot(df):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    repos, repo_display_names, data, box_colors = prepare_boxplot_data(df)

    bp = ax.boxplot(
        data,
        tick_labels=repo_display_names,
        patch_artist=True,
        widths=0.6,
        # whis=[0, 100],  # Extend whiskers to min/max instead of 1.5*IQR
        showfliers=False,
        showmeans=True,
        medianprops=dict(color="black", linewidth=1.2),
        meanprops=dict(
            marker="x",
            markerfacecolor="none",
            markeredgecolor="indigo",
            markersize=5,
        ),
        # flierprops=dict(
        #     marker="o",
        #     markerfacecolor="none",
        #     markeredgecolor="gray",
        #     markersize=3,
        #     alpha=0.5,
        # ),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.text(
        0.02,
        0.98,
        "“×” represents the mean resolution time",
        transform=ax.transAxes,
        fontsize=FONT_SIZES["annotation"],
        color="indigo",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

    ax.set_ylabel("Issue Resolution Time (Days)", fontsize=FONT_SIZES["axis_label"])
    plt.xticks(rotation=45, ha="right", fontsize=FONT_SIZES["tick"])
    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)
    return fig


def save_figure(fig, output_dir, figures_dir, base_name, rq_name):
    fig.savefig(output_dir / f"{base_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{rq_name}.pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)


def main():
    df = load_df()
    resolution_df = prepare_resolution_data(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Identify and save zero-day resolution issues for investigation
    identify_and_save_zero_day_issues(resolution_df, OUTPUT_DIR)

    stats = calculate_resolution_statistics(resolution_df)
    save_statistics_table(stats, OUTPUT_DIR)

    fig = plot_resolution_time_boxplot(resolution_df)
    save_figure(
        fig,
        OUTPUT_DIR,
        FIGURES_DIR,
        "resolution_time_boxplot",
        "rq4_resolution_time_boxplot",
    )


if __name__ == "__main__":
    main()
