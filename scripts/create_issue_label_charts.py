from constants import DATA_DIR
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from plot_utils import (
    PLOT_LINE_WIDTH_THIN,
    setup_plotting_style,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    apply_grid_style,
    FONT_SIZES,
    get_repo_color_mapping,
)

LABELS = [
    "Bug",
    "Infrastructure",
    "Agent Related Issues & Implementations",
    "Data Processing",
    "Documentation",
    "Feature",
    "Community Engagement",
]

LABELS_TO_DISPLAY_FORMAT = {
    "Bug": "Bug",
    "Infrastructure": "Infrastructure",
    "Agent Related Issues & Implementations": "Agent Issues",
    "Data Processing": "Data Processing",
    "Documentation": "Documentation",
    "Feature": "Feature",
    "Community Engagement": "Community",
}


def load_df():
    df = pd.read_parquet(DATA_DIR / "issues_with_categorized_labels_nocutoff.parquet")
    return df


def filter_issues_for_analysis(df):
    filtered_df = df[
        df["categorized_labels"].apply(
            lambda labels: any(label in LABELS for label in labels)
        )
    ].copy()
    return filtered_df


def add_prominent_labels_column(df):
    df["prominent_labels"] = df["categorized_labels"].apply(
        lambda labels: list(set(label for label in labels if label in LABELS))
    )
    return df


def plot_issue_label_trend(df, date_field="created_at"):
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    for label in LABELS:
        label_df = df[
            df["prominent_labels"].apply(lambda labels: label in labels)
        ].copy()
        label_df = label_df.sort_values(date_field)

        # Calculate cumulative count for each issue
        label_df["cumulative"] = range(1, len(label_df) + 1)

        # Add starting point at zero
        start_date = label_df[date_field].iloc[0]
        dates = pd.concat(
            [pd.Series([start_date]), label_df[date_field]], ignore_index=True
        )
        cumulative = pd.concat(
            [pd.Series([0]), pd.Series(label_df["cumulative"].values)],
            ignore_index=True,
        )

        ax.plot(
            dates,
            cumulative,
            label=LABELS_TO_DISPLAY_FORMAT.get(label, label),
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel("Number of Issues", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=0)

    setup_legend(
        ax,
        loc="upper left",
        max_width=15,
    )

    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def plot_new_issues_by_label(df, date_field="created_at"):
    setup_plotting_style()

    df = df.copy()
    if df[date_field].dt.tz is not None:
        df[date_field] = df[date_field].dt.tz_convert(None)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    for label in LABELS:
        label_df = df[
            df["prominent_labels"].apply(lambda labels: label in labels)
        ].copy()

        # Group by date (day) and count issues per day
        daily_counts = (
            label_df.groupby(label_df[date_field].dt.date)
            .size()
            .reset_index(name="count")
        )
        daily_counts.columns = ["date", "count"]
        daily_counts["date"] = pd.to_datetime(daily_counts["date"])

        daily_counts = daily_counts.sort_values("date")

        # Add 30-day rolling average
        daily_counts["smoothed"] = (
            daily_counts["count"].rolling(window=30, center=True, min_periods=1).mean()
        )

        ax.plot(
            daily_counts["date"],
            daily_counts["smoothed"],
            label=LABELS_TO_DISPLAY_FORMAT.get(label, label),
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel(
        "New Issues (Rolling Avg: 30 days)", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.to_datetime("2025-07-31"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=0)

    setup_legend(
        ax,
        loc="upper left",
        max_width=15,
    )

    apply_grid_style(ax, major_alpha=0.6, minor_alpha=0.4)

    return fig


def main():
    df = load_df()
    filtered_df = filter_issues_for_analysis(df)
    df_with_prominent_labels = add_prominent_labels_column(filtered_df)

    outputDir = Path("output")
    figuresDir = Path("figures")

    outputDir.mkdir(parents=True, exist_ok=True)
    figuresDir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_issue_label_trend(
        df_with_prominent_labels, date_field="issue_created_at"
    )
    fig1.savefig(outputDir / "issue_label_trend.png", dpi=300, bbox_inches="tight")
    fig1.savefig(
        figuresDir / "rq3_issue_label_trend.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig1)

    fig2 = plot_new_issues_by_label(
        df_with_prominent_labels, date_field="issue_created_at"
    )
    fig2.savefig(outputDir / "new_issues_by_label.png", dpi=300, bbox_inches="tight")
    fig2.savefig(
        figuresDir / "rq3_new_issues_by_label.pdf", bbox_inches="tight", format="pdf"
    )
    plt.close(fig2)

    print(f"\nPNG files saved in: {outputDir}")
    print(f"PDF files saved in: {figuresDir}")


if __name__ == "__main__":
    main()
