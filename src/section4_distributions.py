import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configuration import PRE_POST_METRICS, get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section4_distributions")
GROUP_ORDER = ["Madrid Before", "Madrid After", "Segovia Before", "Segovia After"]
GROUP_COLORS = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]


def make_metric_df(df, metric_name, pre_col, post_col):
    """Reshape the data to a long format for a specific metric, combining pre and post values with site and time information."""
    before = df[["Participant", "site"]].copy()
    before["group"] = before["site"] + " Before"
    before["value"] = pd.to_numeric(df[pre_col], errors="coerce")
    before["metric"] = metric_name

    after = df[["Participant", "site"]].copy()
    after["group"] = after["site"] + " After"
    after["value"] = pd.to_numeric(df[post_col], errors="coerce")
    after["metric"] = metric_name

    out = pd.concat([before, after], ignore_index=True)
    return out.dropna(subset=["value"])


def plot_metric_distribution(df, metric_name, pre_col, post_col):
    """Plot the distribution of scores for a specific metric by site and time using boxplots and overlaid individual points."""
    plot_df = make_metric_df(df, metric_name, pre_col, post_col)
    data = [plot_df.loc[plot_df["group"] == g, "value"] for g in GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=GROUP_ORDER, widths=0.5, showfliers=False)

    for patch, color in zip(bp["boxes"], GROUP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    for i, values in enumerate(data, start=1):
        x = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(x, values, alpha=0.6, s=25)
        if len(values) > 0:
            ax.scatter(i, np.mean(values), color="black", s=60, marker="D", zorder=3)

    ax.set_title(f"{metric_name}: distribution by site and time", fontsize=13, weight="bold")
    ax.set_ylabel("Observed value")
    ax.set_ylim(0, 7 if metric_name == "Test score" else 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=15)
    plt.tight_layout()
    filename = metric_name.lower().replace(" ", "_") + "_distribution.png"
    plt.savefig(SECTION_OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_distributions(df):
    """Generate distribution plots for all metrics defined in PRE_POST_METRICS."""
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_distribution(df, metric_name, pre_col, post_col)

