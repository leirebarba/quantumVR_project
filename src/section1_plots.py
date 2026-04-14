import matplotlib.pyplot as plt
import pandas as pd
from src.configuration import OUTPUT_DIR, PRE_POST_METRICS


def _mean_se(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    mean = series.mean()
    se = series.std(ddof=1) / (len(series) ** 0.5) if len(series) > 1 else 0
    return mean, se


def plot_test_scores_by_site(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    sites = ["Madrid", "Segovia"]
    x = range(len(sites))
    width = 0.35

    before_means, before_errs = [], []
    after_means, after_errs = [], []

    for site in sites:
        subset = df[df["site"] == site]
        m1, e1 = _mean_se(subset["Score"])
        m2, e2 = _mean_se(subset["Score 2"])
        before_means.append(m1)
        before_errs.append(e1)
        after_means.append(m2)
        after_errs.append(e2)

    ax.bar([i - width/2 for i in x], before_means, width, yerr=before_errs, label="Before", capsize=4)
    ax.bar([i + width/2 for i in x], after_means, width, yerr=after_errs, label="After", capsize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sites)
    ax.set_ylabel("Mean test score")
    ax.set_title("Test scores before vs after")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_scores_by_site.png", dpi=300)
    plt.close()


def plot_metric_grid(long_df):
    metrics = list(PRE_POST_METRICS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = long_df[long_df["metric"] == metric]

        summary = (
            sub.groupby(["site", "time"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["se"] = summary["std"] / summary["count"] ** 0.5

        sites = ["Madrid", "Segovia"]
        x = range(len(sites))
        width = 0.35

        before = summary[summary["time"] == "Before"].set_index("site").reindex(sites)
        after = summary[summary["time"] == "After"].set_index("site").reindex(sites)

        ax.bar([j - width/2 for j in x], before["mean"], width, yerr=before["se"], label="Before", capsize=4)
        ax.bar([j + width/2 for j in x], after["mean"], width, yerr=after["se"], label="After", capsize=4)

        ax.set_xticks(list(x))
        ax.set_xticklabels(sites)
        ax.set_title(metric)
        ax.set_ylim(0, max(7 if metric == "Test score" else 5, after["mean"].max() + 1))

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Pre/post comparison across metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prepost_metric_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_site_time_panels(df):
    """
    Four separate plots:
    Madrid Before, Madrid After, Segovia Before, Segovia After
    using the 5 main metrics.
    """
    metric_labels = list(PRE_POST_METRICS.keys())

    for site in ["Madrid", "Segovia"]:
        for time in ["Before", "After"]:
            means = []
            errors = []

            for metric, (pre_col, post_col) in PRE_POST_METRICS.items():
                col = pre_col if time == "Before" else post_col
                subset = df[df["site"] == site][col]
                m, e = _mean_se(subset)
                means.append(m)
                errors.append(e)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metric_labels, means, yerr=errors, capsize=4)
            ax.set_ylim(0, 7)
            ax.set_ylabel("Mean score")
            ax.set_title(f"{site} - {time}")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            filename = f"{site.lower()}_{time.lower()}_panel.png".replace(" ", "_")
            plt.savefig(OUTPUT_DIR / filename, dpi=300)
            plt.close()
            