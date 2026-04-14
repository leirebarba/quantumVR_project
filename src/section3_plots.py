import matplotlib.pyplot as plt
import pandas as pd
from src.configuration import OUTPUT_DIR


def plot_seminar_comparison(change_df):
    metrics = ["Test score", "Interest", "Relevance", "Future work", "Confidence"]

    for site in ["Madrid", "Segovia"]:
        sub = change_df[change_df["site"] == site].copy()

        attendee_means = []
        non_attendee_means = []

        for metric in metrics:
            attendee_means.append(
                sub.loc[sub["seminar_attendee"], f"{metric}_change"].mean()
            )
            non_attendee_means.append(
                sub.loc[~sub["seminar_attendee"], f"{metric}_change"].mean()
            )

        x = range(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([i - width/2 for i in x], attendee_means, width, label="Attended seminar")
        ax.bar([i + width/2 for i in x], non_attendee_means, width, label="Did not attend")

        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylabel("Mean change (after - before)")
        ax.set_title(f"{site}: seminar attendees vs non-attendees")
        ax.axhline(0, linewidth=1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{site.lower()}_seminar_comparison.png", dpi=300)
        plt.close()
        