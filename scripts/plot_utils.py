import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import numpy as np
from scipy.interpolate import make_interp_spline

# Color palettes
MAIN_COLORS = sns.color_palette("colorblind")
PAIRED_COLORS = sns.color_palette("Paired", n_colors=12)
CATEGORICAL_COLORS = sns.color_palette("Set3", n_colors=10)
DIVERGING_COLORS = sns.color_palette("RdYlBu", n_colors=10)
PIE_COLORS = sns.color_palette("Set2", n_colors=12)
GREY_COLORS_DARK = sns.color_palette("Greys_r", n_colors=10)

# Figure sizes
FIG_SIZE_SINGLE_COL = (3.5, 2.2)
# FIG_SIZE_DOUBLE_COL = (7.16, 4.0)
FIG_SIZE_MEDIUM = (3.5, 3.0)
FIG_SIZE_SMALL = (3.5, 2.0)

# Font sizes
FONT_SIZES = {
    "title": 9.5,
    "axis_label": 8.4,
    "tick": 7.2,
    "legend": 7.2,
    "annotation": 6,
}

# Plot styling constants
PLOT_LINE_WIDTH = 1.2
PLOT_LINE_WIDTH_THIN = 0.7
MARKER_SIZE = 4
MARKER_STYLES = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
GRID_ALPHA = 0.4  # Increased from 0.25 for better visibility
LEGEND_NCOL = 2
TITLE_PAD = 8
AXIS_LABEL_PAD = 4
LAYOUT_PAD = 0.8
MARKER_EDGECOLOR = "w"
MARKER_EDGEWIDTH = 0.3

FONT_FAMILY = "sans-serif"
FONT_WEIGHT_NORMAL = "normal"
FONT_WEIGHT_BOLD = "bold"


def get_repo_color_mapping(repos):
    """Create a consistent color mapping for repositories."""

    repo_color_map = {
        "langchain-ai/langchain": MAIN_COLORS[0],
        "run-llama/llama_index": MAIN_COLORS[1],
        "microsoft/autogen": MAIN_COLORS[2],
        "deepset-ai/haystack": MAIN_COLORS[3],
        "crewAIInc/crewAI": MAIN_COLORS[4],
        "microsoft/semantic-kernel": MAIN_COLORS[5],
        "TransformerOptimus/SuperAGI": MAIN_COLORS[6],
        "letta-ai/letta": MAIN_COLORS[7],
        "FoundationAgents/MetaGPT": MAIN_COLORS[8],
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


def format_time_label(date, granularity):
    if granularity == "week":
        return f"{date.year}\nW{date.strftime('%W')}"
    elif granularity == "month":
        return f"{date.year}\n{date.strftime('%b')}"
    elif granularity == "quarter":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}\nQ{quarter}"
    else:
        return str(date.year)


def setup_plotting_style():
    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "figure.titlesize": FONT_SIZES["title"],
            "figure.facecolor": "white",
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZES["axis_label"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.titleweight": FONT_WEIGHT_BOLD,
            "axes.labelsize": FONT_SIZES["axis_label"],
            "axes.labelweight": FONT_WEIGHT_NORMAL,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 4,  # Add tick marks
            "ytick.major.size": 4,  # Add tick marks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": FONT_SIZES["legend"],
            "legend.title_fontsize": FONT_SIZES["legend"],
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "legend.framealpha": 0.8,
            "legend.borderpad": 0.4,
            "axes.grid": True,
            "grid.alpha": 0.6,
            "grid.color": "#CCCCCC",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "axes.titlepad": 12,
        }
    )


def setup_axis_ticks(ax, dates, granularity, n_ticks=8, rotation=0):
    n_ticks = min(n_ticks if n_ticks else 8, len(dates))
    tick_indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    tick_dates = [dates[i] for i in tick_indices]

    ax.set_xticks(tick_dates)
    formatted_labels = [
        format_time_label(date, granularity).replace(" ", "\n") for date in tick_dates
    ]
    ax.set_xticklabels(
        formatted_labels,
        rotation=rotation,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["tick"],
        color="#333333",
    )

    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=4,
        width=0.8,
        color="#333333",
        top=False,
        bottom=True,
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.6)


def setup_categorical_axis_ticks(ax, labels, n_ticks=None, rotation=0):
    """Similar to setup_axis_ticks but for categorical (non-date) x-axis labels"""
    n_ticks = min(n_ticks if n_ticks else len(labels), len(labels))
    tick_positions = np.arange(len(labels))

    formatted_labels = [label.replace(" ", "\n") for label in labels]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        formatted_labels,
        rotation=rotation,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["tick"],
        color="#333333",
    )

    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=4,
        width=0.8,
        color="#333333",
        top=False,
        bottom=True,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.6)


def setup_legend(
    ax,
    title=None,
    loc="upper left",
    bbox_to_anchor=None,
    ncol=1,
    frameon=True,
    wrap_text=True,
    max_width=20,
    handlelength=1,
    borderpad=0.4,
    columnspacing=1.0,
):
    from textwrap import wrap

    # Get current legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Sort legend entries alphabetically by label to ensure consistent ordering
    if labels:
        try:
            # Pair labels with handles and sort by label (case-insensitive)
            paired = sorted(zip(labels, handles), key=lambda x: x[0].lower())
            labels, handles = zip(*paired)
            labels = list(labels)
            handles = list(handles)
        except Exception:
            # Fallback: keep original order if sorting fails for any reason
            labels = list(labels)
            handles = list(handles)

    # Wrap labels if requested
    if wrap_text:
        labels = ["\n".join(wrap(label, max_width)) for label in labels]

    legend = ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.8,
        borderpad=borderpad,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )

    if title:
        legend.get_title().set_fontsize(FONT_SIZES["legend"])
        legend.get_title().set_color("#333333")
    plt.setp(legend.get_texts(), fontsize=FONT_SIZES["legend"], color="#333333")


def apply_grid_style(ax, major_alpha=0.3, minor_alpha=0.2):
    ax.grid(
        True,
        which="major",
        linestyle="-",  # Solid lines
        alpha=major_alpha,  # Lighter
        color="#CCCCCC",  # Lighter color
        linewidth=0.5,
    )
    ax.grid(
        True,
        which="minor",
        linestyle="-",  # Solid lines
        alpha=minor_alpha,  # Lighter
        color="#DDDDDD",  # Even lighter for minor
        linewidth=0.3,
    )

    ax.set_axisbelow(True)


def create_pie_chart(data, labels, title, figsize=FIG_SIZE_SINGLE_COL, explode=None):
    fig, ax = plt.subplots(figsize=figsize)

    if explode is None:
        explode = [0.01] * len(data)

    wedges, texts, autotexts = ax.pie(
        data,
        explode=explode,
        labels=labels,
        colors=PIE_COLORS,
        autopct="%1.1f%%",
        pctdistance=0.85,
        wedgeprops=dict(width=0.7),
    )

    plt.setp(
        autotexts,
        color="black",
        weight=FONT_WEIGHT_BOLD,
        fontsize=FONT_SIZES["annotation"],
    )
    plt.setp(texts, fontsize=FONT_SIZES["annotation"], weight=FONT_WEIGHT_NORMAL)

    ax.set_title(title, pad=12, fontsize=FONT_SIZES["title"], weight=FONT_WEIGHT_BOLD)

    return fig, ax


def save_plot(fig, output_path, base_filename, dpi=600):
    fig.savefig(output_path / f"{base_filename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path / f"{base_filename}.pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)


def smooth_with_bspline(dates, values, n_points=300):
    x = np.array([(d - dates[0]).total_seconds() for d in dates])
    y = values.copy()

    x_smooth = np.linspace(x.min(), x.max(), n_points)

    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    dates_smooth = [dates[0] + timedelta(seconds=int(xs)) for xs in x_smooth]

    return dates_smooth, y_smooth
