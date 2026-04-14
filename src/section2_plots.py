import matplotlib.pyplot as plt
import pandas as pd
from src.configuration import OUTPUT_DIR, VR_METRICS


def plot_vr_feedback(df):
    vr_df = df[df[VR_METRICS[0]].notna()].copy()

    summary = (
        vr_df.groupby("site")[VR_METRICS]
        .mean(numeric_only=True)
        .T
    )

    short_labels = [
        "Understanding",
        "Motivation",
        "Difficulty",
        "Recommendation"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax)

    ax.set_xticklabels(short_labels, rotation=0)
    ax.set_title("VR Feedback by Site", fontsize=13, weight="bold")
    ax.set_ylabel("Mean Likert Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Site", frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vr_feedback_by_site.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_vr_feedback_overall(df):
    vr_df = df[df[VR_METRICS[0]].notna()].copy()
    means = vr_df[VR_METRICS].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(means.index, means.values)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean Likert score")
    ax.set_title("Overall VR feedback")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vr_feedback_overall.png", dpi=300)
    plt.close()
