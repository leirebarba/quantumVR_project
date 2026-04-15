import math

import matplotlib.pyplot as plt
import pandas as pd

from src.configuration import PRE_POST_METRICS, get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section6_distributions")
GROUP_ORDER = [
    ("Madrid", "Before"),
    ("Madrid", "After"),
    ("Segovia", "Before"),
    ("Segovia", "After"),
]

def _metric_limits(metric_name):
    if metric_name == "Test score":
        return 0, 7, range(0, 8)
    return 1, 5, range(1, 6)


def _build_distribution_df(df, metric_name, pre_col, post_col):
    """Reshape the data to a long format for a specific metric, combining pre and post values with site and time information."""
    before = df[["Participant", "site"]].copy()
    before["time"] = "Before"
    before["value"] = pd.to_numeric(df[pre_col], errors="coerce")
    before["metric"] = metric_name

    after = df[["Participant", "site"]].copy()
    after["time"] = "After"
    after["value"] = pd.to_numeric(df[post_col], errors="coerce")
    after["metric"] = metric_name

    out = pd.concat([before, after], ignore_index=True)
    return out.dropna(subset=["value"])


def plot_metric_histograms(df, metric_name, pre_col, post_col):
    """Plot histograms of scores for a specific metric by site and time, arranged in a 2x2 grid."""
    plot_df = _build_distribution_df(df, metric_name, pre_col, post_col)
    min_val, max_val, tick_values = _metric_limits(metric_name)
    bins = [x - 0.5 for x in range(min_val, max_val + 2)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (site, time) in zip(axes, GROUP_ORDER):
        sub = plot_df[(plot_df["site"] == site) & (plot_df["time"] == time)]["value"]
        ax.hist(sub, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_title(f"{site} - {time}")
        ax.set_xlim(min_val - 0.5, max_val + 0.5)
        ax.set_xticks(list(tick_values))
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{metric_name}: score distribution by site and time", fontsize=14, weight="bold")
    fig.supxlabel("Observed value")
    fig.supylabel("Number of participants")
    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_") + "_histograms.png"
    plt.savefig(SECTION_OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_histograms(df):
    """Generate histogram plots for all metrics defined in PRE_POST_METRICS."""
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_histograms(df, metric_name, pre_col, post_col)
