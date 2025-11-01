from constants import DATA_DIR
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

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

        # Group by month and count issues per month
        label_df["year_month"] = label_df[date_field].dt.to_period("M")
        monthly_counts = label_df.groupby("year_month").size().reset_index(name="count")
        monthly_counts["date"] = monthly_counts["year_month"].dt.to_timestamp()
        monthly_counts = monthly_counts.sort_values("date")

        # Apply PCHIP interpolation for smooth curves
        x = np.array(
            [
                (d - monthly_counts["date"].iloc[0]).total_seconds()
                for d in monthly_counts["date"]
            ]
        )
        y = monthly_counts["count"].values

        # Create PCHIP interpolator
        pchip = PchipInterpolator(x, y)

        # Generate smooth curve
        x_smooth = np.linspace(x.min(), x.max(), min(300, len(x) * 3))
        y_smooth = pchip(x_smooth)

        # Ensure no negative values
        y_smooth = np.maximum(y_smooth, 0)

        # Convert back to dates
        dates_smooth = [
            monthly_counts["date"].iloc[0] + pd.Timedelta(seconds=float(xs))
            for xs in x_smooth
        ]

        ax.plot(
            dates_smooth,
            y_smooth,
            label=LABELS_TO_DISPLAY_FORMAT.get(label, label),
            linewidth=PLOT_LINE_WIDTH_THIN,
        )

    ax.set_ylabel("New Issues", fontsize=FONT_SIZES["axis_label"])
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
