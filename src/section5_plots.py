import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.configuration import OUTPUT_DIR, PRE_POST_METRICS


def make_metric_df(df, metric_name, pre_col, post_col):
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


def mean_se(x):
    x = pd.Series(x).dropna()
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return mean, se


def plot_metric_dots(df, metric_name, pre_col, post_col):
    plot_df = make_metric_df(df, metric_name, pre_col, post_col)

    group_order = ["Madrid Before", "Madrid After", "Segovia Before", "Segovia After"]
    colors = {
        "Madrid Before": "#4C72B0",
        "Madrid After": "#4C72B0",
        "Segovia Before": "#DD8452",
        "Segovia After": "#DD8452",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, group in enumerate(group_order, start=1):
        vals = plot_df.loc[plot_df["group"] == group, "value"].dropna()

        # jittered raw points
        x = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, alpha=0.45, s=35, color=colors[group])

        # mean and SE
        m, se = mean_se(vals)
        ax.errorbar(
            i, m, yerr=se,
            fmt="o",
            color="black",
            capsize=4,
            markersize=8,
            linewidth=1.5,
            zorder=3
        )

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(group_order, rotation=15)
    ax.set_title(f"{metric_name}: individual values by site and time", fontsize=14, weight="bold")
    ax.set_ylabel("Observed value")

    if metric_name == "Test score":
        ax.set_ylim(0, 7)
    else:
        ax.set_ylim(1, 5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    filename = metric_name.lower().replace(" ", "_") + "_dots.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_dots(df):
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_dots(df, metric_name, pre_col, post_col)
