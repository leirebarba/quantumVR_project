import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    out = out.dropna(subset=["value"])
    return out


def plot_metric_distribution(df, metric_name, pre_col, post_col):
    plot_df = make_metric_df(df, metric_name, pre_col, post_col)

    group_order = ["Madrid Before", "Madrid After", "Segovia Before", "Segovia After"]
    data = [plot_df.loc[plot_df["group"] == g, "value"] for g in group_order]

    fig, ax = plt.subplots(figsize=(9, 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=group_order,
        widths=0.5,
        showfliers=False
    )

    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    for i, values in enumerate(data, start=1):
        x = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(x, values, alpha=0.6, s=25)

        mean_val = np.mean(values)
        ax.scatter(i, mean_val, color="black", s=60, marker="D", zorder=3)

    ax.set_title(f"{metric_name}: distribution by site and time", fontsize=13, weight="bold")
    ax.set_ylabel("Observed value")

    if metric_name == "Test score":
        ax.set_ylim(0, 7)
    else:
        ax.set_ylim(1, 5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_") + "_distribution.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_distributions(df):
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_distribution(df, metric_name, pre_col, post_col)

